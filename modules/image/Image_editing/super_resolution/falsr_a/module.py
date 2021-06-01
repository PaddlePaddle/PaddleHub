# -*- coding:utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import os
import argparse

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from falsr_a.data_feed import reader
from falsr_a.processor import postprocess, base64_to_cv2, cv2_to_base64, check_dir


@moduleinfo(
    name="falsr_a",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="falsr_a is a super resolution model.",
    version="1.0.0")
class Falsr_A(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "falsr_a_model")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        self.model_file_path = self.default_pretrained_model_path
        cpu_config = AnalysisConfig(self.model_file_path)
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
            gpu_config = AnalysisConfig(self.model_file_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def reconstruct(self, images=None, paths=None, use_gpu=False, visualization=False, output_dir="falsr_a_output"):
        """
        API for super resolution.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C], the color space is BGR.
            paths (list[str]): The paths of images.
            use_gpu (bool): Whether to use gpu.
            visualization (bool): Whether to save image or not.
            output_dir (str): The path to store output images.

        Returns:
            res (list[dict]): each element in the list is a dict, the keys and values are:
                save_path (str, optional): the path to save images. (Exists only if visualization is True)
                data (numpy.ndarray): data of post processed image.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        res = list()

        for i in range(total_num):
            image_y = np.array([all_data[i]['img_y']])
            image_scale_pbpr = np.array([all_data[i]['img_scale_pbpr']])
            image_y = PaddleTensor(image_y.copy())
            image_scale_pbpr = PaddleTensor(image_scale_pbpr.copy())
            output = self.gpu_predictor.run([image_y, image_scale_pbpr]) if use_gpu else self.cpu_predictor.run(
                [image_y, image_scale_pbpr])
            output = np.expand_dims(output[0].as_ndarray(), axis=1)
            out = postprocess(
                data_out=output,
                org_im=all_data[i]['org_im'],
                org_im_shape=all_data[i]['org_im_shape'],
                org_im_path=all_data[i]['org_im_path'],
                output_dir=output_dir,
                visualization=visualization)
            res.append(out)
        return res

    def save_inference_model(self,
                             dirname='falsr_a_save_model',
                             model_filename=None,
                             params_filename=None,
                             combined=False):
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
        results = self.reconstruct(images=images_decode, **kwargs)
        results = [{'data': cv2_to_base64(result['data'])} for result in results]
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
        results = self.reconstruct(
            paths=[args.input_path], use_gpu=args.use_gpu, output_dir=args.output_dir, visualization=args.visualization)
        if args.save_dir is not None:
            check_dir(args.save_dir)
            self.save_inference_model(args.save_dir)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='falsr_a_output', help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--save_dir', type=str, default='falsr_a_save_model', help="The directory to save model.")
        self.arg_config_group.add_argument(
            '--visualization', type=ast.literal_eval, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")


if __name__ == "__main__":
    module = Falsr_A()
    module.reconstruct(paths=["BSD100_001.png", "BSD100_002.png", "Set5_003.png"])
    module.save_inference_model()
