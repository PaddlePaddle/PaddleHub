# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
import argparse
import copy

import paddle
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable, serving
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import rescale, resize

from .model import FirstOrderPredictor


@moduleinfo(
    name="first_order_motion", type="CV/gan", author="paddlepaddle", author_email="", summary="", version="1.0.0")
class FirstOrderMotion:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "vox-cpk.pdparams")
        self.network = FirstOrderPredictor(weight_path=self.pretrained_model, face_enhancement=True)

    def generate(self,
                 source_image=None,
                 driving_video=None,
                 ratio=0.4,
                 image_size=256,
                 output_dir='./motion_driving_result/',
                 filename='result.mp4',
                 use_gpu=False):
        '''
        source_image (str): path to image<br/>
        driving_video (str) : path to driving_video<br/>
        ratio: margin ratio
        image_size: size of image
        output_dir: the dir to save the results
        filename: filename to save the results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        '''
        paddle.disable_static()
        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)
        if source_image == None or driving_video == None:
            print('No image or driving video provided. Please input an image and a driving video.')
            return
        self.network.run(source_image, driving_video, ratio, image_size, output_dir, filename)

    @runnable
    def run_cmd(self, argvs: list):
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
        self.args = self.parser.parse_args(argvs)
        self.generate(
            source_image=self.args.source_image,
            driving_video=self.args.driving_video,
            ratio=self.args.ratio,
            image_size=self.args.image_size,
            output_dir=self.args.output_dir,
            use_gpu=self.args.use_gpu)
        return

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='motion_driving_result', help='output directory for saving result.')
        self.arg_config_group.add_argument("--filename", default='result.mp4', help="filename to output")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument("--source_image", type=str, help="path to source image")
        self.arg_input_group.add_argument("--driving_video", type=str, help="path to driving video")
        self.arg_input_group.add_argument("--ratio", dest="ratio", type=float, default=0.4, help="margin ratio")
        self.arg_input_group.add_argument(
            "--image_size", dest="image_size", type=int, default=256, help="size of image")
