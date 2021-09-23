# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import os
import argparse

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from paddle.inference import Config
from paddle.inference import create_predictor

from paddlehub.module.module import moduleinfo, runnable, serving

from human_pose_estimation_resnet50_mpii.processor import base64_to_cv2, postprocess
from human_pose_estimation_resnet50_mpii.data_feed import reader
from human_pose_estimation_resnet50_mpii.pose_resnet import ResNet


@moduleinfo(
    name="human_pose_estimation_resnet50_mpii",
    type="CV/keypoint_detection",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.comi",
    summary=
    "Paddle implementation for the paper `Simple baselines for human pose estimation and tracking`, trained with the MPII dataset.",
    version="1.1.1")
class HumanPoseEstimation(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "pose-resnet50-mpii-384x384")
        self._set_config()

    def _get_device_id(self, places):
        try:
            places = os.environ[places]
            id = int(places)
        except:
            id = -1
        return id

    def _set_config(self):
        """
        predictor config setting
        """

        # create default cpu predictor
        cpu_config = Config(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        # create predictors using various types of devices

        # npu
        npu_id = self._get_device_id("FLAGS_selected_npus")
        if npu_id != -1:
            # use npu
            npu_config = Config(self.default_pretrained_model_path)
            npu_config.disable_glog_info()
            npu_config.enable_npu(device_id=npu_id)
            self.npu_predictor = create_predictor(npu_config)

        # gpu
        gpu_id = self._get_device_id("CUDA_VISIBLE_DEVICES")
        if gpu_id != -1:
            # use gpu
            gpu_config = Config(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=gpu_id)
            self.gpu_predictor = create_predictor(gpu_config)

        # xpu
        xpu_id = self._get_device_id("XPU_VISIBLE_DEVICES")
        if xpu_id != -1:
            # use xpu
            xpu_config = Config(self.default_pretrained_model_path)
            xpu_config.disable_glog_info()
            xpu_config.enable_xpu(100)
            self.xpu_predictor = create_predictor(xpu_config)

    def keypoint_detection(self,
                           images=None,
                           paths=None,
                           batch_size=1,
                           use_gpu=False,
                           output_dir='output_pose',
                           visualization=False,
                           use_device=None):
        """
        API for human pose estimation and tracking.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C].
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            use_device (str): use cpu, gpu, xpu or npu, overwrites use_gpu flag.

        Returns:
            res (list[dict]): each element of res is a dict, keys contains 'path', 'data', the corresponding valus are:
                path (str): the path of original image.
                data (OrderedDict): The key points of human pose.
        """

        # real predictor to use
        if use_device is not None:
            if use_device == "cpu":
                predictor = self.cpu_predictor
            elif use_device == "xpu":
                predictor = self.xpu_predictor
            elif use_device == "npu":
                predictor = self.npu_predictor
            elif use_device == "gpu":
                predictor = self.gpu_predictor
            else:
                raise Exception("Unsupported device: " + use_device)
        else:
            # use_device is not set, therefore follow use_gpu
            if use_gpu:
                predictor = self.gpu_predictor
            else:
                predictor = self.cpu_predictor

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

            input_names = predictor.get_input_names()
            input_tensor = predictor.get_input_handle(input_names[0])
            input_tensor.reshape(batch_image.shape)
            input_tensor.copy_from_cpu(batch_image.copy())
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            predictor_output = output_handle.copy_to_cpu()

            output = np.expand_dims(predictor_output, axis=1)

            # postprocess one by one
            for i in range(len(batch_data)):
                out = postprocess(out_heatmaps=output[i],
                                  org_im=batch_data[i]['org_im'],
                                  org_im_shape=batch_data[i]['org_im_shape'],
                                  org_im_path=batch_data[i]['org_im_path'],
                                  output_dir=output_dir,
                                  visualization=visualization)
                res.append(out)
        return res

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.default_pretrained_model_path, executor=exe)

        fluid.io.save_inference_model(dirname=dirname,
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
        results = self.keypoint_detection(images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the human_pose_estimation_resnet50_mpii module.",
                                              prog='hub run human_pose_estimation_resnet50_mpii',
                                              usage='%(prog)s',
                                              add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.keypoint_detection(paths=[args.input_path],
                                          batch_size=args.batch_size,
                                          use_gpu=args.use_gpu,
                                          output_dir=args.output_dir,
                                          visualization=args.visualization,
                                          use_device=args.use_device)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU or not")
        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='output_pose',
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether to save output as images.")
        self.arg_config_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")
        self.arg_config_group.add_argument('--use_device',
                                           choices=["cpu", "gpu", "xpu", "npu"],
                                           help="use cpu, gpu, xpu or npu. overwrites use_gpu flag.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
