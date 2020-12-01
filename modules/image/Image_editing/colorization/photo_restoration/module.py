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
import paddle.nn as nn
import paddlehub as hub
from paddlehub.module.module import moduleinfo, serving, Module

import photo_restoration.utils as U


@moduleinfo(
    name="photo_restoration",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="photo_restoration is a photo restoration model based on deoldify and realsr.",
    version="1.0.0")
class PhotoRestoreModel(Module):
    """
    PhotoRestoreModel

    Args:
        load_checkpoint(str): Checkpoint save path, default is None.
        visualization (bool): Whether to save the estimation result. Default is True.
    """

    def _initialize(self, visualization: bool = False):
        #super(PhotoRestoreModel, self).__init__()
        self.deoldify = hub.Module(name='deoldify')
        self.realsr = hub.Module(name='realsr')
        self.visualization = visualization

    def run_image(self,
                  input,
                  model_select: list = ['Colorization', 'SuperResolution'],
                  save_path: str = 'photo_restoration'):
        self.models = []
        for model in model_select:
            print('\n {} model proccess start..'.format(model))
            if model == 'Colorization':
                self.deoldify.eval()
                self.models.append(self.deoldify)
            if model == 'SuperResolution':
                self.realsr.eval()
                self.models.append(self.realsr)

        for model in self.models:
            output = model.run_image(input)
            input = output
        if self.visualization:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img_name = str(time.time()) + '.png'
            save_img = os.path.join(save_path, img_name)
            cv2.imwrite(save_img, output)
            print("save result at: ", save_img)

        return output

    @serving
    def serving_method(self, images, model_select):
        """
        Run as a service.
        """
        print(model_select)
        images_decode = U.base64_to_cv2(images)
        results = self.run_image(input=images_decode, model_select=model_select)
        results = U.cv2_to_base64(results)
        return results
