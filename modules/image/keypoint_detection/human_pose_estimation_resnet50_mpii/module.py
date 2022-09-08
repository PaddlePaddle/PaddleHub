# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import os
import argparse

import paddle
import paddle.jit
import paddle.static
import numpy as np
from paddle.inference import Config, create_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from .processor import base64_to_cv2, postprocess
from .data_feed import reader


@moduleinfo(
    name="human_pose_estimation_resnet50_mpii",
    type="CV/keypoint_detection",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.comi",
    summary=
    "Paddle implementation for the paper `Simple baselines for human pose estimation and tracking`, trained with the MPII dataset.",
    version="1.2.0")
class HumanPoseEstimation:
    def __init__(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "pose-resnet50-mpii-384x384", "model")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        model = self.default_pretrained_model_path+'.pdmodel'
        params = self.default_pretrained_model_path+'.pdiparams'
        cpu_config = Config(model, params)
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
            gpu_config = Config(model, params)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_predictor(gpu_config)

    def keypoint_detection(self,
                           images=None,
                           paths=None,
                           batch_size=1,
                           use_gpu=False,
                           output_dir='output_pose',
                           visualization=False):
        """
        API for human pose estimation and tracking.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C].
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.

        Returns:
            res (list[dict]): each element of res is a dict, keys contains 'path', 'data', the corresponding valus are:
                path (str): the path of original image.
                data (OrderedDict): The key points of human pose.
        """
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
            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(batch_image)
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            output = np.expand_dims(output_handle.copy_to_cpu(), axis=1)
            # postprocess one by one
            for i in range(len(batch_data)):
                out = postprocess(
                    out_heatmaps=output[i],
                    org_im=batch_data[i]['org_im'],
                    org_im_shape=batch_data[i]['org_im_shape'],
                    org_im_path=batch_data[i]['org_im_path'],
                    output_dir=output_dir,
                    visualization=visualization)
                res.append(out)
        return res

    def save_inference_model(self, path):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        program, feed_target_names, fetch_targets = paddle.static.load_inference_model(self.default_pretrained_model_path, exe)
        global_block = program.global_block()
        feed_vars = [global_block.var(item) for item in feed_target_names]
        paddle.static.save_inference_model(
            path,
            feed_vars=feed_vars,
            fetch_vars=fetch_targets,
            executor=exe,
            program=program
        )

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
        self.parser = argparse.ArgumentParser(
            description="Run the human_pose_estimation_resnet50_mpii module.",
            prog='hub run human_pose_estimation_resnet50_mpii',
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.keypoint_detection(
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
            '--output_dir', type=str, default='output_pose', help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=ast.literal_eval, default=False, help="whether to save output as images.")
        self.arg_config_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
