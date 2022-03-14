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
import os

import cv2
import numpy as np
import paddle

import paddlehub as hub
from .enlighten_inference.pd_model.x2paddle_code import ONNXModel
from .util import base64_to_cv2
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="enlightengan",
            type="CV/enlighten",
            author="paddlepaddle",
            author_email="",
            summary="",
            version="1.0.0")
class EnlightenGAN:

    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "enlighten_inference/pd_model")
        self.model = ONNXModel()
        params = paddle.load(os.path.join(self.pretrained_model, 'model.pdparams'))
        self.model.set_dict(params, use_structured_name=True)

    def enlightening(self,
                     images: list = None,
                     paths: list = None,
                     output_dir: str = './enlightening_result/',
                     use_gpu: bool = False,
                     visualization: bool = True):
        '''
        enlighten images in the low-light scene.

        images (list[numpy.ndarray]): data of images, shape of each is [H, W, C], color space must be BGR(read by cv2).
        paths (list[str]): paths to images
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
        self.model.eval()

        if images != None:
            for image in images:
                image = image[:, :, ::-1]
                image = np.expand_dims(np.transpose(image, (2, 0, 1)).astype(np.float32) / 255., 0)
                inputtensor = paddle.to_tensor(image)
                out, out1 = self.model(inputtensor)
                out = out.numpy()[0]
                out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
                out = np.clip(out, 0, 255)
                out = out.astype('uint8')
                results.append(out)

        if paths != None:
            for path in paths:
                image = cv2.imread(path)[:, :, ::-1]
                image = np.expand_dims(np.transpose(image, (2, 0, 1)).astype(np.float32) / 255., 0)
                inputtensor = paddle.to_tensor(image)
                out, out1 = self.model(inputtensor)
                out = out.numpy()[0]
                out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
                out = np.clip(out, 0, 255)
                out = out.astype('uint8')
                results.append(out)

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
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        self.args = self.parser.parse_args(argvs)
        results = self.enlightening(paths=[self.args.input_path],
                                    output_dir=self.args.output_dir,
                                    use_gpu=self.args.use_gpu,
                                    visualization=self.args.visualization)
        return results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.enlightening(images=images_decode, **kwargs)
        tolist = [result.tolist() for result in results]
        return tolist

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='enlightening_result',
                                           help='output directory for saving result.')
        self.arg_config_group.add_argument('--visualization', type=bool, default=False, help='save results or not.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to input image.")
