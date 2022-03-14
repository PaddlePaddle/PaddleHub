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
import argparse
import copy
import os

import cv2
import numpy as np
import paddle
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import resize

import paddlehub as hub
from .model import LapStylePredictor
from .util import base64_to_cv2
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(
    name="lapstyle_ocean",
    type="CV/style_transfer",
    author="paddlepaddle",
    author_email="",
    summary="",
    version="1.0.0")
class Lapstyle_ocean:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "lapstyle_ocean.pdparams")

        self.network = LapStylePredictor(weight_path=self.pretrained_model)

    def style_transfer(self,
                       images: list = None,
                       paths: list = None,
                       output_dir: str = './transfer_result/',
                       use_gpu: bool = False,
                       visualization: bool = True):
        '''
        Transfer a image to ocean style.

        images (list[dict]): data of images, each element is a dict:
          - content (numpy.ndarray): input image，shape is \[H, W, C\]，BGR format；<br/>
          - style (numpy.ndarray) : style image，shape is \[H, W, C\]，BGR format；<br/>
        paths (list[dict]): paths to images, eacg element is a dict:
          - content (str): path to input image；<br/>
          - style (str) : path to style image；<br/>

         output_dir (str): the dir to save the results
         use_gpu (bool): if True, use gpu to perform the computation, otherwise cpu.
         visualization (bool): if True, save results in output_dir.
        '''
        results = []
        paddle.disable_static()
        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)
        if images == None and paths == None:
            print('No image provided. Please input an image or a image path.')
            return

        if images != None:
            for image_dict in images:
                content_img = image_dict['content']
                style_img = image_dict['style']
                results.append(self.network.run(content_img, style_img))

        if paths != None:
            for path_dict in paths:
                content_img = cv2.imread(path_dict['content'])
                style_img = cv2.imread(path_dict['style'])
                results.append(self.network.run(content_img, style_img))

        if visualization == True:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            for i, out in enumerate(results):
                cv2.imwrite(os.path.join(output_dir, 'output_{}.png'.format(i)), out[:, :, ::-1])

        return results

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

        self.style_transfer(
            paths=[{
                'content': self.args.content,
                'style': self.args.style
            }],
            output_dir=self.args.output_dir,
            use_gpu=self.args.use_gpu,
            visualization=self.args.visualization)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = copy.deepcopy(images)
        for image in images_decode:
            image['content'] = base64_to_cv2(image['content'])
            image['style'] = base64_to_cv2(image['style'])
        results = self.style_transfer(images_decode, **kwargs)
        tolist = [result.tolist() for result in results]
        return tolist

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='transfer_result', help='output directory for saving result.')
        self.arg_config_group.add_argument('--visualization', type=bool, default=False, help='save results or not.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--content', type=str, help="path to content image.")
        self.arg_input_group.add_argument('--style', type=str, help="path to style image.")
