# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import argparse
import os

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from paddle.inference import Config
from paddle.inference import create_predictor

from paddlehub.module.module import moduleinfo, runnable, serving

from pyramidbox_lite_server_mask.data_feed import reader
from pyramidbox_lite_server_mask.processor import postprocess, base64_to_cv2


@moduleinfo(
    name="pyramidbox_lite_server_mask",
    type="CV/face_detection",
    author="baidu-vis",
    author_email="",
    summary=
    "PyramidBox-Lite-Server-Mask is a high-performance face detection model used to detect whether people wear masks.",
    version="1.3.0")
class PyramidBoxLiteServerMask(hub.Module):
    def _initialize(self, face_detector_module=None):
        """
        Args:
            face_detector_module (class): module to detect face.
        """
        self.default_pretrained_model_path = os.path.join(self.directory, "pyramidbox_lite_server_mask_model")
        if face_detector_module is None:
            self.face_detector = hub.Module(name='pyramidbox_lite_server')
        else:
            self.face_detector = face_detector_module
        self._set_config()
        self.processor = self

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

    def set_face_detector_module(self, face_detector_module):
        """
        Set face detector.
        Args:
            face_detector_module (class): module to detect faces.
        """
        self.face_detector = face_detector_module

    def get_face_detector_module(self):
        return self.face_detector

    def face_detection(self,
                       images=None,
                       paths=None,
                       data=None,
                       batch_size=1,
                       use_gpu=False,
                       visualization=False,
                       output_dir='detection_result',
                       use_multi_scale=False,
                       shrink=0.5,
                       confs_threshold=0.6,
                       use_device=None):
        """
        API for face detection.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C], color space must be BGR.
            paths (list[str]): The paths of images.
            use_gpu (bool): Whether to use gpu.
            visualization (bool): Whether to save image or not.
            output_dir (str): The path to store output images.
            use_multi_scale (bool): whether to enable multi-scale face detection. Enabling multi-scale face detection
                can increase the accuracy to detect faces, however,
                it reduce the prediction speed for the increase model calculation.
            shrink (float): parameter to control the resize scale in preprocess.
            confs_threshold (float): confidence threshold.
            use_device (str): use cpu, gpu, xpu or npu, overwrites use_gpu flag.

        Returns:
            res (list[dict]): The result of face detection and save path of images.
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

        # compatibility with older versions
        if data:
            if 'image' in data:
                if paths is None:
                    paths = list()
                paths += data['image']
            elif 'data' in data:
                if images is None:
                    images = list()
                images += data['data']

        # get all data
        all_element = list()
        for yield_data in reader(self.face_detector, shrink, confs_threshold, images, paths, use_gpu, use_multi_scale,
                                 use_device):
            all_element.append(yield_data)

        image_list = list()
        element_image_num = list()
        for i in range(len(all_element)):
            element_image = [handled['image'] for handled in all_element[i]['preprocessed']]
            element_image_num.append(len(element_image))
            image_list.extend(element_image)

        total_num = len(image_list)
        loop_num = int(np.ceil(total_num / batch_size))

        predict_out = np.zeros((1, 2))
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for element_id in range(batch_size):
                try:
                    batch_data.append(image_list[handle_id + element_id])
                except:
                    pass

            batch_image = np.squeeze(np.array(batch_data), axis=1)

            input_names = predictor.get_input_names()
            input_tensor = predictor.get_input_handle(input_names[0])
            input_tensor.reshape(batch_image.shape)
            input_tensor.copy_from_cpu(batch_image.copy())
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            predictor_output = output_handle.copy_to_cpu()

            predict_out = np.concatenate((predict_out, predictor_output))

        predict_out = predict_out[1:]
        # postprocess one by one
        res = list()
        for i in range(len(all_element)):
            detect_faces_list = [handled['face'] for handled in all_element[i]['preprocessed']]
            interval_left = sum(element_image_num[0:i])
            interval_right = interval_left + element_image_num[i]
            out = postprocess(confidence_out=predict_out[interval_left:interval_right],
                              org_im=all_element[i]['org_im'],
                              org_im_path=all_element[i]['org_im_path'],
                              detected_faces=detect_faces_list,
                              output_dir=output_dir,
                              visualization=visualization)
            res.append(out)
        return res

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        classifier_dir = os.path.join(dirname, 'mask_detector')
        detector_dir = os.path.join(dirname, 'pyramidbox_lite')
        self._save_classifier_model(classifier_dir, model_filename, params_filename, combined)
        self._save_detector_model(detector_dir, model_filename, params_filename, combined)

    def _save_detector_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        self.face_detector.save_inference_model(dirname, model_filename, params_filename, combined)

    def _save_classifier_model(self, dirname, model_filename=None, params_filename=None, combined=True):
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
        results = self.face_detection(images_decode, **kwargs)
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
        results = self.face_detection(paths=[args.input_path],
                                      use_gpu=args.use_gpu,
                                      output_dir=args.output_dir,
                                      visualization=args.visualization,
                                      shrink=args.shrink,
                                      confs_threshold=args.confs_threshold,
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
                                           default='detection_result',
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
        self.arg_input_group.add_argument(
            '--shrink',
            type=ast.literal_eval,
            default=0.5,
            help="resize the image to `shrink * original_shape` before feeding into network.")
        self.arg_input_group.add_argument('--confs_threshold',
                                          type=ast.literal_eval,
                                          default=0.6,
                                          help="confidence threshold.")
