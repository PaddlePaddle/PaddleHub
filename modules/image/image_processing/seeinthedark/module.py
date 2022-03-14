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
from paddlehub.module.module import moduleinfo, runnable, serving
import numpy as np
import rawpy
import cv2

from .util import base64_to_cv2


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw
    if not isinstance(raw, np.ndarray):
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
        self.cpu_have_loaded = False
        self.gpu_have_loaded = False

    def set_device(self, use_gpu=False):
        if use_gpu == False:
            if not self.cpu_have_loaded:
                exe = paddle.static.Executor(paddle.CPUPlace())
                [prog, inputs, outputs] = paddle.static.load_inference_model(
                    path_prefix=self.pretrained_model,
                    executor=exe,
                    model_filename="model.pdmodel",
                    params_filename="model.pdiparams")
                self.cpuexec, self.cpuprog, self.cpuinputs, self.cpuoutputs = exe, prog, inputs, outputs
                self.cpu_have_loaded = True

            return self.cpuexec, self.cpuprog, self.cpuinputs, self.cpuoutputs

        else:
            if not self.gpu_have_loaded:
                exe = paddle.static.Executor(paddle.CUDAPlace(0))
                [prog, inputs, outputs] = paddle.static.load_inference_model(
                    path_prefix=self.pretrained_model,
                    executor=exe,
                    model_filename="model.pdmodel",
                    params_filename="model.pdiparams")
                self.gpuexec, self.gpuprog, self.gpuinputs, self.gpuoutputs = exe, prog, inputs, outputs
                self.gpu_have_loaded = True

            return self.gpuexec, self.gpuprog, self.gpuinputs, self.gpuoutputs

    def denoising(self,
                  images: list = None,
                  paths: list = None,
                  output_dir: str = './enlightening_result/',
                  use_gpu: bool = False,
                  visualization: bool = True):
        '''
        Denoise a raw image in the low-light scene.

        images (list[numpy.ndarray]): data of images, shape of each is [H, W], must be sing-channel image captured by camera.
        paths (list[str]): paths to images
        output_dir: the dir to save the results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        visualization: if True, save results in output_dir.
        '''
        results = []
        paddle.enable_static()
        exe, prog, inputs, outputs = self.set_device(use_gpu)

        if images != None:
            for raw in images:
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
                output = output * 255
                output = np.clip(output, 0, 255)
                output = output.astype('uint8')
                results.append(output)
        if paths != None:
            for path in paths:
                raw = rawpy.imread(path)
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
                output = output * 255
                output = np.clip(output, 0, 255)
                output = output.astype('uint8')
                results.append(output)

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
        self.denoising(
            paths=[self.args.input_path],
            output_dir=self.args.output_dir,
            use_gpu=self.args.use_gpu,
            visualization=self.args.visualization)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.denoising(images=images_decode, **kwargs)
        tolist = [result.tolist() for result in results]
        return tolist

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='denoising_result', help='output directory for saving result.')
        self.arg_config_group.add_argument('--visualization', type=bool, default=False, help='save results or not.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to input raw image, should be raw file captured by camera.")
