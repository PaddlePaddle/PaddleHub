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

import paddle
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable
from PIL import Image
import numpy as np
from .enlighten_inference import EnlightenOnnxModel
from .enlighten_inference.pd_model.x2paddle_code import ONNXModel


@moduleinfo(
    name="EnlightenGAN", type="CV/enlighten", author="paddlepaddle", author_email="", summary="", version="1.0.0")
class EnlightenGAN:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "enlighten_inference/pd_model")

    def enlightening(self, input_path, output_path='./enlightening_result.png', use_gpu=False):
        '''
        enlighten a image in the low-light scene.

        input_path: the image path
        output_path: the path to save the results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        '''
        paddle.disable_static()
        img = np.array(Image.open(input_path))
        img = np.expand_dims(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255., 0)
        inputtensor = paddle.to_tensor(img)
        params = paddle.load(os.path.join(self.pretrained_model, 'model.pdparams'))
        model = ONNXModel()
        model.set_dict(params, use_structured_name=True)
        model.eval()
        out, out1 = model(inputtensor)
        out = out.numpy()[0]
        out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
        out = np.clip(out, 0, 255)
        out = out.astype('uint8')

        print('enlighten Over.')
        try:
            Image.fromarray(out).save(os.path.join(output_path))
            print('Image saved in {}'.format(output_path))
        except:
            print('Save image failed. Please check the output_path, should\
                be image format ext, e.g. png. current output path {}'.format(output_path))
        return out

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
        self.enlightening(input_path=self.args.input_path, output_path=self.args.output_path, use_gpu=self.args.use_gpu)

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_path', type=str, default='enlightening_result.png', help='output path for saving result.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to input image.")
