# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import numpy as np
from PIL import Image

import paddlehub.io.augmentation as image_augmentation

color_mode_dict = {
    "RGB": [0, 1, 2],
    "RBG": [0, 2, 1],
    "GBR": [1, 2, 0],
    "GRB": [1, 0, 2],
    "BGR": [2, 1, 0],
    "BRG": [2, 0, 1]
}


class ImageClassificationReader(object):
    def __init__(self,
                 image_width,
                 image_height,
                 dataset,
                 color_mode="RGB",
                 data_augmentation=False):
        self.image_width = image_width
        self.image_height = image_height
        self.color_mode = color_mode
        self.dataset = dataset
        self.data_augmentation = data_augmentation
        if self.color_mode not in color_mode_dict:
            raise ValueError(
                "Color_mode should in %s." % color_mode_dict.keys())

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image width and height should not be negative.")

    def data_generator(self, batch_size, phase="train", shuffle=False):
        if phase == "train":
            data = self.dataset.train_data(shuffle)
        elif phase == "test":
            shuffle = False
            data = self.dataset.test_data(shuffle)
        elif phase == "val" or phase == "dev":
            shuffle = False
            data = self.dataset.validate_data(shuffle)

        def _data_reader():
            for image_path, label in data:
                image = Image.open(image_path)
                image = image_augmentation.image_resize(image, self.image_width,
                                                        self.image_height)
                if self.data_augmentation:
                    image = image_augmentation.image_random_process(
                        image, enable_resize=False)

                # only support RGB
                image = image.convert('RGB')

                # HWC to CHW
                image = np.array(image)
                if len(image.shape) == 3:
                    image = np.swapaxes(image, 1, 2)
                    image = np.swapaxes(image, 1, 0)

                image = image[color_mode_dict[self.color_mode], :, :]
                yield ((image, label))

        return paddle.batch(_data_reader, batch_size=batch_size)
