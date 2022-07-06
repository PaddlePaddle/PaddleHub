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
from yolov3_darknet53_pedestrian.data_feed import reader
from yolov3_darknet53_pedestrian.processor import base64_to_cv2
from yolov3_darknet53_pedestrian.processor import load_label_info
from yolov3_darknet53_pedestrian.processor import postprocess

import paddlehub as hub
from paddlehub.common.paddle_helper import add_vars_prefix
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="yolov3_darknet53_pedestrian",
            version="1.0.3",
            type="CV/object_detection",
            summary="Baidu's YOLOv3 model for pedestrian detection, with backbone DarkNet53.",
            author="paddlepaddle",
            author_email="paddle-dev@baidu.com")
class YOLOv3DarkNet53Pedestrian(hub.Module):

    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "yolov3_darknet53_pedestrian_model")
        self.label_names = load_label_info(os.path.join(self.directory, "label_file.txt"))
        self._set_config()

    def _set_config(self):
        """
        predictor config setting.
        """
        cpu_config = Config(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_ir_optim(False)
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
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_predictor(gpu_config)

    def object_detection(self,
                         paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='yolov3_pedestrian_detect_output',
                         score_thresh=0.2,
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
            res (list[dict]): The result of pedestrian detecion. keys include 'data', 'save_path', the corresponding value is:
                data (dict): the result of object detection, keys include 'left', 'top', 'right', 'bottom', 'label', 'confidence', the corresponding value is:
                    left (float): The X coordinate of the upper left corner of the bounding box;
                    top (float): The Y coordinate of the upper left corner of the bounding box;
                    right (float): The X coordinate of the lower right corner of the bounding box;
                    bottom (float): The Y coordinate of the lower right corner of the bounding box;
                    label (str): The label of detection result;
                    confidence (float): The confidence of detection result.
                save_path (str, optional): The path to save output images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Attempt to use GPU for prediction, but environment variable CUDA_VISIBLE_DEVICES was not set correctly."
                )

        paths = paths if paths else list()
        data_reader = partial(reader, paths, images)
        batch_reader = paddle.batch(data_reader, batch_size=batch_size)
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            feed_data = np.array(feed_data)

            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(np.array(list(feed_data[:, 0])))
            input_handle = predictor.get_input_handle(input_names[1])
            input_handle.copy_from_cpu(np.array(list(feed_data[:, 1])))

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
                                        score_thresh=args.score_thresh)
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
                                           default='yolov3_pedestrian_detect_output',
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
        self.arg_input_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")
        self.arg_input_group.add_argument('--score_thresh',
                                          type=ast.literal_eval,
                                          default=0.2,
                                          help="threshold for object detecion.")
