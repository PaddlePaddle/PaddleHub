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

from ultra_light_fast_generic_face_detector_1mb_320.processor import postprocess, base64_to_cv2
from ultra_light_fast_generic_face_detector_1mb_320.data_feed import reader


@moduleinfo(
    name="ultra_light_fast_generic_face_detector_1mb_320",
    type="CV/face_detection",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    summary=
    "Ultra-Light-Fast-Generic-Face-Detector-1MB is a high-performance object detection model release on https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB.",
    version="1.1.2")
class FaceDetector320(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory,
                                                          "ultra_light_fast_generic_face_detector_1mb_320")
        self._set_config()

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

    def face_detection(self,
                       images=None,
                       paths=None,
                       data=None,
                       batch_size=1,
                       use_gpu=False,
                       output_dir='face_detector_320_predict_output',
                       visualization=False,
                       confs_threshold=0.5,
                       iou_threshold=0.5):
        """
        API for face detection.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C], color space is BGR.
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            confs_threshold (float): threshold for confidence coefficient.
            iou_threshold (float): threshold for iou.

        Returns:
            res (list[dict()]): The result of face detection and save path of images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        # compatibility with older versions
        if data and 'image' in data:
            if paths is None:
                paths = []
            paths += data['image']

        # get all data
        all_data = []
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = []
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
            batch_image = PaddleTensor(batch_image.astype('float32'))
            data_out = self.gpu_predictor.run([batch_image]) if use_gpu else self.cpu_predictor.run([batch_image])
            confidences = data_out[0].as_ndarray()
            boxes = data_out[1].as_ndarray()

            # postprocess one by one
            for i in range(len(batch_data)):
                out = postprocess(
                    confidences=confidences[i],
                    boxes=boxes[i],
                    orig_im=batch_data[i]['orig_im'],
                    orig_im_shape=batch_data[i]['orig_im_shape'],
                    orig_im_path=batch_data[i]['orig_im_path'],
                    output_dir=output_dir,
                    visualization=visualization,
                    confs_threshold=confs_threshold,
                    iou_threshold=iou_threshold)
                res.append(out)
        return res

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
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.face_detection(
            paths=[args.input_path],
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir',
            type=str,
            default='face_detector_320_predict_output',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=ast.literal_eval, default=False, help="whether to save output as images.")
        self.arg_config_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
