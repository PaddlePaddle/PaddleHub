# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import argparse
import ast
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import paddle
from face_landmark_localization.data_feed import reader
from face_landmark_localization.processor import base64_to_cv2
from face_landmark_localization.processor import postprocess
from paddle.inference import Config
from paddle.inference import create_predictor

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(
    name="face_landmark_localization",
    type="CV/keypoint_detection",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    summary=
    "Face_Landmark_Localization can be used to locate face landmark. This Module is trained through the MPII Human Pose dataset.",
    version="1.0.3")
class FaceLandmarkLocalization(hub.Module):

    def _initialize(self, face_detector_module=None):
        """
        Args:
            face_detector_module (class): module to detect face.
        """
        self.default_pretrained_model_path = os.path.join(self.directory, "face_landmark_localization")
        if face_detector_module is None:
            self.face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
        else:
            self.face_detector = face_detector_module
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = Config(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = Config(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_predictor(gpu_config)

    def set_face_detector_module(self, face_detector_module):
        """
        Set face detector.

        Args:
            face_detector_module (class): module to detect face.
        """
        self.face_detector = face_detector_module

    def get_face_detector_module(self):
        return self.face_detector

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = paddle.CPUPlace()
        exe = paddle.Executor(place)

        program, feeded_var_names, target_vars = paddle.static.load_inference_model(
            dirname=self.default_pretrained_model_path, executor=exe)
        face_landmark_dir = os.path.join(dirname, "face_landmark")
        detector_dir = os.path.join(dirname, "detector")

        paddle.static.save_inference_model(dirname=face_landmark_dir,
                                           main_program=program,
                                           executor=exe,
                                           feeded_var_names=feeded_var_names,
                                           target_vars=target_vars,
                                           model_filename=model_filename,
                                           params_filename=params_filename)
        self.face_detector.save_inference_model(dirname=detector_dir,
                                                model_filename=model_filename,
                                                params_filename=params_filename,
                                                combined=combined)

    def keypoint_detection(self,
                           images=None,
                           paths=None,
                           batch_size=1,
                           use_gpu=False,
                           output_dir='face_landmark_output',
                           visualization=False):
        """
        API for face landmark.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C].
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.

        Returns:
            res (list[dict()]): The key points of face landmark and save path of images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        # get all data
        all_data = []
        for yield_data in reader(self.face_detector, images, paths, use_gpu):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = []
        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['face'] for data in batch_data]).astype('float32')
            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(batch_image)

            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            points = output_handle.copy_to_cpu()

            for idx, sample in enumerate(batch_data):
                sample['points'] = points[idx].reshape(68, -1)
            res += batch_data

        res = postprocess(res, output_dir, visualization)
        return res

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
        results = self.keypoint_detection(paths=[args.input_path],
                                          use_gpu=args.use_gpu,
                                          output_dir=args.output_dir,
                                          visualization=args.visualization)
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
                                           default=None,
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
