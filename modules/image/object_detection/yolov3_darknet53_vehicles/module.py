# coding=utf-8
from __future__ import absolute_import

import argparse
import ast
import os
from functools import partial

import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from yolov3_darknet53_vehicles.data_feed import reader
from yolov3_darknet53_vehicles.processor import base64_to_cv2
from yolov3_darknet53_vehicles.processor import load_label_info
from yolov3_darknet53_vehicles.processor import postprocess

import paddlehub as hub
from paddlehub.common.paddle_helper import add_vars_prefix
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="yolov3_darknet53_vehicles",
            version="1.0.3",
            type="CV/object_detection",
            summary="Baidu's YOLOv3 model for vehicles detection, with backbone DarkNet53.",
            author="paddlepaddle",
            author_email="paddle-dev@baidu.com")
class YOLOv3DarkNet53Vehicles(hub.Module):

    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "yolov3_darknet53_vehicles_model")
        self.label_names = load_label_info(os.path.join(self.directory, "label_file.txt"))
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
        predictor config setting.
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

    def object_detection(self,
                         paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='yolov3_vehicles_detect_output',
                         score_thresh=0.2,
                         visualization=True,
                         use_device=None):
        """API of Object Detection.

        Args:
            paths (list[str]): The paths of images.
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            score_thresh (float): threshold for object detecion.
            use_device (str): use cpu, gpu, xpu or npu, overwrites use_gpu flag.

        Returns:
            res (list[dict]): The result of vehicles detecion. keys include 'data', 'save_path', the corresponding value is:
                data (dict): the result of object detection, keys include 'left', 'top', 'right', 'bottom', 'label', 'confidence', the corresponding value is:
                    left (float): The X coordinate of the upper left corner of the bounding box;
                    top (float): The Y coordinate of the upper left corner of the bounding box;
                    right (float): The X coordinate of the lower right corner of the bounding box;
                    bottom (float): The Y coordinate of the lower right corner of the bounding box;
                    label (str): The label of detection result;
                    confidence (float): The confidence of detection result.
                save_path (str, optional): The path to save output images.
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

        paths = paths if paths else list()
        data_reader = partial(reader, paths, images)
        batch_reader = paddle.batch(data_reader, batch_size=batch_size)
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            feed_data = np.array(feed_data)

            input_names = predictor.get_input_names()
            image_data = np.array(list(feed_data[:, 0]))
            image_size_data = np.array(list(feed_data[:, 1]))

            image_tensor = predictor.get_input_handle(input_names[0])
            image_tensor.reshape(image_data.shape)
            image_tensor.copy_from_cpu(image_data.copy())

            image_size_tensor = predictor.get_input_handle(input_names[1])
            image_size_tensor.reshape(image_size_data.shape)
            image_size_tensor.copy_from_cpu(image_size_data.copy())

            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])

            output = postprocess(paths=paths,
                                 images=images,
                                 data_out=output_handle,
                                 score_thresh=score_thresh,
                                 label_names=self.label_names,
                                 output_dir=output_dir,
                                 handle_id=iter_id * batch_size,
                                 visualization=visualization)
            res.extend(output)
        return res

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = paddle.CPUPlace()
        exe = paddle.Executor(place)

        program, feeded_var_names, target_vars = paddle.static.load_inference_model(
            dirname=self.default_pretrained_model_path, executor=exe)

        paddle.static.save_inference_model(dirname=dirname,
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
        results = self.object_detection(images=images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.object_detection(paths=[args.input_path],
                                        batch_size=args.batch_size,
                                        use_gpu=args.use_gpu,
                                        output_dir=args.output_dir,
                                        visualization=args.visualization,
                                        score_thresh=args.score_thresh,
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
                                           default='yolov3_vehicles_detect_output',
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether to save output as images.")
        self.arg_config_group.add_argument('--use_device',
                                           choices=["cpu", "gpu", "xpu", "npu"],
                                           help="use cpu, gpu, xpu or npu. overwrites use_gpu flag.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
        self.arg_input_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")
        self.arg_input_group.add_argument('--score_thresh',
                                          type=ast.literal_eval,
                                          default=0.2,
                                          help="threshold for object detecion.")
