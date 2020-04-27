# coding=utf-8
from __future__ import absolute_import

import ast
import argparse
import os
from functools import partial

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving
from paddlehub.common.paddle_helper import add_vars_prefix

from yolov3_mobilenet_v1_coco2017.mobilenet_v1 import MobileNet
from yolov3_mobilenet_v1_coco2017.processor import load_label_info, postprocess, base64_to_cv2
from yolov3_mobilenet_v1_coco2017.data_feed import reader
from yolov3_mobilenet_v1_coco2017.yolo_head import MultiClassNMS, YOLOv3Head


@moduleinfo(
    name="yolov3_mobilenet_v1_coco2017",
    version="1.1.0",
    type="CV/object_detection",
    summary=
    "Baidu's YOLOv3 model for object detection with backbone MobileNet_V1, trained with dataset COCO2017.",
    author="paddlepaddle",
    author_email="")
class YOLOv3MobileNetV1Coco2017(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "yolov3_mobilenet_v1_model")
        self.label_names = load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self._set_config()

    def _set_config(self):
        """
        predictor config setting.
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_ir_optim(False)
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
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(self,
                trainable=True,
                pretrained=True,
                var_prefix='',
                get_prediction=False):
        """
        Distill the Head Features, so as to perform transfer learning.

        Args:
            trainable (bool): whether to set parameters trainable.
            pretrained (bool): whether to load default pretrained model.
            var_prefix (str): prefix to append to the varibles.
            get_prediction (bool): whether to get prediction.

        Returns:
             inputs(dict): the input variables.
             outputs(dict): the output variables.
             context_prog (Program): the program to execute transfer learning.
        """
        context_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(context_prog, startup_program):
            with fluid.unique_name.guard():
                # image
                image = fluid.layers.data(
                    name='image', shape=[3, 608, 608], dtype='float32')
                # backbone
                backbone = MobileNet(
                    norm_type='sync_bn',
                    norm_decay=0.,
                    conv_group_scale=1,
                    with_extra_blocks=False)
                # body_feats
                body_feats = backbone(image)
                # im_size
                im_size = fluid.layers.data(
                    name='im_size', shape=[2], dtype='int32')
                # yolo_head
                yolo_head = YOLOv3Head(num_classes=80)
                # head_features
                head_features = yolo_head._get_outputs(
                    body_feats, is_train=trainable)

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(startup_program)

                # var_prefix
                var_prefix = var_prefix if var_prefix else '@HUB_{}@'.format(
                    self.name)
                # name of inputs
                inputs = {
                    'image': var_prefix + image.name,
                    'im_size': var_prefix + im_size.name
                }
                # name of outputs
                if get_prediction:
                    bbox_out = yolo_head.get_prediction(head_features, im_size)
                    outputs = {'bbox_out': [var_prefix + bbox_out.name]}
                else:
                    outputs = {
                        'head_features':
                        [var_prefix + var.name for var in head_features]
                    }
                # add_vars_prefix
                add_vars_prefix(context_prog, var_prefix)
                add_vars_prefix(startup_program, var_prefix)
                # inputs
                inputs = {
                    key: context_prog.global_block().vars[value]
                    for key, value in inputs.items()
                }
                # outputs
                outputs = {
                    key: [
                        context_prog.global_block().vars[varname]
                        for varname in value
                    ]
                    for key, value in outputs.items()
                }
                # trainable
                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    fluid.io.load_vars(
                        exe,
                        self.default_pretrained_model_path,
                        predicate=_if_exist)
                else:
                    exe.run(startup_program)

                return inputs, outputs, context_prog

    def object_detection(self,
                         paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='detection_result',
                         score_thresh=0.5,
                         visualization=True):
        """API of Object Detection.

        Args:
            paths (list[str]): The paths of images.
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            score_thresh (float): threshold for object detecion.

        Returns:
            res (list[dict]): The result of coco2017 detecion. keys include 'data', 'save_path', the corresponding value is:
                data (dict): the result of object detection, keys include 'left', 'top', 'right', 'bottom', 'label', 'confidence', the corresponding value is:
                    left (float): The X coordinate of the upper left corner of the bounding box;
                    top (float): The Y coordinate of the upper left corner of the bounding box;
                    right (float): The X coordinate of the lower right corner of the bounding box;
                    bottom (float): The Y coordinate of the lower right corner of the bounding box;
                    label (str): The label of detection result;
                    confidence (float): The confidence of detection result.
                save_path (str, optional): The path to save output images.
        """
        paths = paths if paths else list()
        data_reader = partial(reader, paths, images)
        batch_reader = fluid.io.batch(data_reader, batch_size=batch_size)
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            feed_data = np.array(feed_data)
            image_tensor = PaddleTensor(np.array(list(feed_data[:, 0])))
            im_size_tensor = PaddleTensor(np.array(list(feed_data[:, 1])))
            if use_gpu:
                data_out = self.gpu_predictor.run(
                    [image_tensor, im_size_tensor])
            else:
                data_out = self.cpu_predictor.run(
                    [image_tensor, im_size_tensor])

            output = postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_dir,
                handle_id=iter_id * batch_size,
                visualization=visualization)
            res.extend(output)
        return res

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
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
        results = self.face_detection(images_decode, **kwargs)
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
        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.face_detection(
            paths=[args.input_path],
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization,
            score_thresh=args.score_thresh)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir',
            type=str,
            default='detection_result',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=False,
            help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to image.")
        self.arg_input_group.add_argument(
            '--batch_size',
            type=ast.literal_eval,
            default=1,
            help="batch size.")
        self.arg_input_group.add_argument(
            '--score_thresh',
            type=ast.literal_eval,
            default=0.5,
            help="threshold for object detecion.")
