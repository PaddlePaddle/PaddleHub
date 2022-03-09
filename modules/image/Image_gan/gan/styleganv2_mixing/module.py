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

from .model import StyleGANv2MixingPredictor
from .util import base64_to_cv2


@moduleinfo(
    name="styleganv2_mixing",
    type="CV/style_transfer",
    author="paddlepaddle",
    author_email="",
    summary="",
    version="1.0.0")
class styleganv2_mixing:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "stylegan2-ffhq-config-f.pdparams")
        self.network = StyleGANv2MixingPredictor(weight_path=self.pretrained_model, model_type='ffhq-config-f')
        self.pixel2style2pixel_module = hub.Module(name='pixel2style2pixel')

    def generate(self,
                 images=None,
                 paths=None,
                 weights=[0.5] * 18,
                 output_dir='./mixing_result/',
                 use_gpu=False,
                 visualization=True):
        '''
        images (list[dict]): data of images, each element is a dict，the keys are as below：
          - image1 (numpy.ndarray): image1 to be mixed，shape is \[H, W, C\]，BGR format；<br/>
          - image2 (numpy.ndarray) : image2 to be mixed，shape is \[H, W, C\]，BGR format；<br/>
        paths (list[str]): paths to images, each element is a dict，the keys are as below：
          - image1 (str): path to image1；<br/>
          - image2 (str) : path to image2；<br/>
        weights (list(float)): weight for mixing
        output_dir: the dir to save the results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        visualization: if True, save results in output_dir.
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
                image1 = image_dict['image1'][:, :, ::-1]
                image2 = image_dict['image2'][:, :, ::-1]
                _, latent1 = self.pixel2style2pixel_module.network.run(image1)
                _, latent2 = self.pixel2style2pixel_module.network.run(image2)
                results.append(self.network.run(latent1, latent2, weights))

        if paths != None:
            for path_dict in paths:
                path1 = path_dict['image1']
                path2 = path_dict['image2']
                image1 = cv2.imread(path1)[:, :, ::-1]
                image2 = cv2.imread(path2)[:, :, ::-1]
                _, latent1 = self.pixel2style2pixel_module.network.run(image1)
                _, latent2 = self.pixel2style2pixel_module.network.run(image2)
                results.append(self.network.run(latent1, latent2, weights))

        if visualization == True:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            for i, out in enumerate(results):
                if out is not None:
                    cv2.imwrite(os.path.join(output_dir, 'src_{}_image1.png'.format(i)), out[0][:, :, ::-1])
                    cv2.imwrite(os.path.join(output_dir, 'src_{}_image2.png'.format(i)), out[1][:, :, ::-1])
                    cv2.imwrite(os.path.join(output_dir, 'dst_{}.png'.format(i)), out[2][:, :, ::-1])

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
        results = self.generate(
            paths=[{
                'image1': self.args.image1,
                'image2': self.args.image2
            }],
            weights=self.args.weights,
            output_dir=self.args.output_dir,
            use_gpu=self.args.use_gpu,
            visualization=self.args.visualization)
        return results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = copy.deepcopy(images)
        for image in images_decode:
            image['image1'] = base64_to_cv2(image['image1'])
            image['image2'] = base64_to_cv2(image['image2'])
        results = self.generate(images_decode, **kwargs)
        tolist = [result.tolist() for result in results]
        return tolist

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='mixing_result', help='output directory for saving result.')
        self.arg_config_group.add_argument('--visualization', type=bool, default=False, help='save results or not.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--image1', type=str, help="path to input image1.")
        self.arg_input_group.add_argument('--image2', type=str, help="path to input image2.")
        self.arg_input_group.add_argument(
            "--weights",
            type=float,
            nargs="+",
            default=[0.5] * 18,
            help="different weights at each level of two latent codes")
