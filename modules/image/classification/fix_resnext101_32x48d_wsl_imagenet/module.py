# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import argparse
import os

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving
from paddlehub.common.paddle_helper import add_vars_prefix

from fix_resnext101_32x48d_wsl_imagenet.processor import postprocess, base64_to_cv2
from fix_resnext101_32x48d_wsl_imagenet.data_feed import reader
from fix_resnext101_32x48d_wsl_imagenet.resnext101_wsl import Fix_ResNeXt101_32x48d_wsl


@moduleinfo(
    name="fix_resnext101_32x48d_wsl_imagenet",
    type="CV/image_classification",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    summary="fix_resnext101_32x48d_wsl is a image classfication model, this module is trained with imagenet datasets.",
    version="1.0.0")
class FixResnext10132x48dwslImagenet(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "model")
        label_file = os.path.join(self.directory, "label_list.txt")
        with open(label_file, 'r', encoding='utf-8') as file:
            self.label_list = file.read().split("\n")[:-1]
        self.predictor_set = False

    def get_expected_image_width(self):
        return 224

    def get_expected_image_height(self):
        return 224

    def get_pretrained_images_mean(self):
        im_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3)
        return im_mean

    def get_pretrained_images_std(self):
        im_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3)
        return im_std

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_paddle_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = AnalysisConfig(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(self, trainable=True, pretrained=True):
        """context for transfer learning.

        Args:
            trainable (bool): Set parameters in program to be trainable.
            pretrained (bool) : Whether to load pretrained model.

        Returns:
            inputs (dict): key is 'image', corresponding vaule is image tensor.
            outputs (dict): key is :
                'classification', corresponding value is the result of classification.
                'feature_map', corresponding value is the result of the layer before the fully connected layer.
            context_prog (fluid.Program): program for transfer learning.
        """
        context_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(context_prog, startup_prog):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name="image", shape=[3, 224, 224], dtype="float32")
                resnet_vd = Fix_ResNeXt101_32x48d_wsl()
                output, feature_map = resnet_vd.net(input=image, class_dim=len(self.label_list))

                name_prefix = '@HUB_{}@'.format(self.name)
                inputs = {'image': name_prefix + image.name}
                outputs = {'classification': name_prefix + output.name, 'feature_map': name_prefix + feature_map.name}
                add_vars_prefix(context_prog, name_prefix)
                add_vars_prefix(startup_prog, name_prefix)
                global_vars = context_prog.global_block().vars
                inputs = {key: global_vars[value] for key, value in inputs.items()}
                outputs = {key: global_vars[value] for key, value in outputs.items()}

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        b = os.path.exists(os.path.join(self.default_pretrained_model_path, var.name))
                        return b

                    fluid.io.load_vars(exe, self.default_pretrained_model_path, context_prog, predicate=_if_exist)
                else:
                    exe.run(startup_prog)
                # trainable
                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable
        return inputs, outputs, context_prog

    def classification(self, images=None, paths=None, batch_size=1, use_gpu=False, top_k=1):
        """
        API for image classification.

        Args:
            images (list[numpy.ndarray]): data of images, shape of each is [H, W, C], color space must be BGR.
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            top_k (int): Return top k results.

        Returns:
            res (list[dict]): The classfication results.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        if not self.predictor_set:
            self._set_config()
            self.predictor_set = True

        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = list()
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['image'] for data in batch_data])
            batch_image = PaddleTensor(batch_image.copy())
            predictor_output = self.gpu_predictor.run([batch_image]) if use_gpu else self.cpu_predictor.run(
                [batch_image])
            out = postprocess(data_out=predictor_output[0].as_ndarray(), label_list=self.label_list, top_k=top_k)
            res += out
        return res

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.default_pretrained_model_path, executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
            main_program=program,
            executor=exe,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            model_filename=model_filename,
            params_filename=params_filename)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.classification(images=images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
            usage='%(prog)s',
            add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.classification(paths=[args.input_path], batch_size=args.batch_size, use_gpu=args.use_gpu)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU or not.")
        self.arg_config_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")
        self.arg_config_group.add_argument('--top_k', type=ast.literal_eval, default=1, help="Return top k results.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
