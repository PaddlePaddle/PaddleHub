# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import os
import time

import cv2
import paddle
import paddle.nn as nn
import paddlehub as hub
from paddlehub.module.module import moduleinfo, serving, Module


@moduleinfo(name="video_restoration",
            type="CV/image_editing",
            author="paddlepaddle",
            author_email="",
            summary="video_restoration is a video restoration model based on dain, deoldify and edvr.",
            version="1.0.0")
class PhotoRestoreModel(Module):
    """
    PhotoRestoreModel

    Args:
        output_path(str): Path to save results.

    """
    def _initialize(self, output_path='output'):
        self.output_path = output_path
        paddle.enable_static()
        self.dain = hub.Module(name='dain', output_path=self.output_path)
        self.edvr = hub.Module(name='edvr', output_path=self.output_path)
        paddle.disable_static()
        self.deoldify = hub.Module(name='deoldify', output_path=self.output_path)

    def predict(self,
                input_video_path,
                model_select=['Interpolation', 'Colorization', 'SuperResolution']):
        temp_video_path = None
        for model in model_select:
            print('\n {} model proccess start..'.format(model))
            if model == 'Interpolation':
                paddle.enable_static()
                print('dain input:',input_video_path)
                frames_path, temp_video_path = self.dain.predict(input_video_path)
                input_video_path = temp_video_path
                paddle.disable_static()
            
            if model == 'Colorization':
                self.deoldify.eval()
                print('deoldify input:',input_video_path)
                frames_path, temp_video_path = self.deoldify.predict(input_video_path)
                input_video_path = temp_video_path
                
            if model == 'SuperResolution':
                paddle.enable_static()
                print('edvr input:',input_video_path)
                frames_path, temp_video_path = self.edvr.predict(input_video_path)
                input_video_path = temp_video_path
                paddle.disable_static()

        return temp_video_path


