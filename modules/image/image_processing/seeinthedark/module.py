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
import numpy as np
import rawpy
from PIL import Image


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :], im[0:H:2, 1:W:2, :], im[1:H:2, 1:W:2, :], im[1:H:2, 0:W:2, :]), axis=2)
    return out


@moduleinfo(
    name="seeinthedark", type="CV/denoising", author="paddlepaddle", author_email="", summary="", version="1.0.0")
class LearningToSeeInDark:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "pd_model/inference_model")

    def denoising(self, input_path, output_path='./denoising_result.png', use_gpu=False):
        '''
        Denoise a raw image in the low-light scene.

        input_path: the raw image path
        output_path: the path to save the results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        '''
        paddle.enable_static()
        if use_gpu == False:
            exe = paddle.static.Executor(paddle.CPUPlace())
        else:
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
        [prog, inputs, outputs] = paddle.static.load_inference_model(
            path_prefix=self.pretrained_model,
            executor=exe,
            model_filename="model.pdmodel",
            params_filename="model.pdiparams")
        raw = rawpy.imread(input_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * 300
        px = input_full.shape[1] // 512
        py = input_full.shape[2] // 512
        rx, ry = px * 512, py * 512
        input_full = input_full[:, :rx, :ry, :]
        output = np.random.randn(rx * 2, ry * 2, 3)
        input_full = np.minimum(input_full, 1.0)
        for i in range(px):
            for j in range(py):
                input_patch = input_full[:, i * 512:i * 512 + 512, j * 512:j * 512 + 512, :]
                result = exe.run(prog, feed={inputs[0]: input_patch}, fetch_list=outputs)
                output[i * 512 * 2:i * 512 * 2 + 512 * 2, j * 512 * 2:j * 512 * 2 + 512 * 2, :] = result[0][0]
        output = np.minimum(np.maximum(output, 0), 1)

        print('Denoising Over.')
        try:
            Image.fromarray(np.uint8(output * 255)).save(os.path.join(output_path))
            print('Image saved in {}'.format(output_path))
        except:
            print('Save image failed. Please check the output_path, should\
                be image format ext, e.g. png. current output path {}'.format(output_path))

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
        self.denoising(input_path=self.args.input_path, output_path=self.args.output_path, use_gpu=self.args.use_gpu)

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_path', type=str, default='denoising_result.png', help='output path for saving result.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to input raw image, should be raw file captured by camera.")
