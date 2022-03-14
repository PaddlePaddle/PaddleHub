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

from .model import Wav2LipPredictor


@moduleinfo(name="wav2lip", type="CV/generation", author="paddlepaddle", author_email="", summary="", version="1.0.0")
class wav2lip:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "wav2lip_hq.pdparams")

        self.network = Wav2LipPredictor(
            checkpoint_path=self.pretrained_model,
            static=False,
            fps=25,
            pads=[0, 10, 0, 0],
            face_det_batch_size=16,
            wav2lip_batch_size=128,
            resize_factor=1,
            crop=[0, -1, 0, -1],
            box=[-1, -1, -1, -1],
            rotate=False,
            nosmooth=False,
            face_detector='sfd',
            face_enhancement=True)

    def wav2lip_transfer(self, face, audio, output_dir='./output_result/', use_gpu=False, visualization=True):
        '''
        face (str): path to video/image that contains faces to use.
        audio (str): path to input audio.
        output_dir: the dir to save the results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        visualization: if True, save results in output_dir.
        '''
        paddle.disable_static()
        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)
        self.network.run(face, audio, output_dir, visualization)

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
        self.wav2lip_transfer(
            face=self.args.face,
            audio=self.args.audio,
            output_dir=self.args.output_dir,
            use_gpu=self.args.use_gpu,
            visualization=self.args.visualization)
        return

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='output_result', help='output directory for saving result.')
        self.arg_config_group.add_argument('--visualization', type=bool, default=False, help='save results or not.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--audio', type=str, help="path to input audio.")
        self.arg_input_group.add_argument('--face', type=str, help="path to video/image that contains faces to use.")
