#coding:utf-8
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

channel_order_dict = {
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
                 channel_order="RGB",
                 images_mean=None,
                 images_std=None,
                 data_augmentation=False):
        self.image_width = image_width
        self.image_height = image_height
        self.channel_order = channel_order
        self.dataset = dataset
        self.data_augmentation = data_augmentation
        self.images_std = images_std
        self.images_mean = images_mean

        if self.images_mean is None:
            try:
                self.images_mean = self.dataset.images_mean
            except:
                self.images_mean = [0, 0, 0]
        self.images_mean = np.array(self.images_mean).reshape(3, 1, 1)

        if self.images_std is None:
            try:
                self.images_std = self.dataset.images_std
            except:
                self.images_std = [1, 1, 1]
        self.images_std = np.array(self.images_std).reshape(3, 1, 1)

        if self.channel_order not in channel_order_dict:
            raise ValueError(
                "The channel_order should in %s." % channel_order_dict.keys())

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image width and height should not be negative.")

    def data_generator(self,
                       batch_size,
                       phase="train",
                       shuffle=False,
                       data=None):
        if phase == "train":
            data = self.dataset.train_data(shuffle)
        elif phase == "test":
            shuffle = False
            data = self.dataset.test_data(shuffle)
        elif phase == "val" or phase == "dev":
            shuffle = False
            data = self.dataset.validate_data(shuffle)
        elif phase == "predict":
            data = data

        def preprocess(image_path):
            image = Image.open(image_path)
            image = image_augmentation.image_resize(image, self.image_width,
                                                    self.image_height)
            if self.data_augmentation:
                image = image_augmentation.image_random_process(
                    image, enable_resize=False, enable_crop=False)

            # only support RGB
            image = image.convert('RGB')

            # HWC to CHW
            image = np.array(image).astype('float32')
            if len(image.shape) == 3:
                image = np.swapaxes(image, 1, 2)
                image = np.swapaxes(image, 1, 0)

            # standardization
            image /= 255
            image -= self.images_mean
            image /= self.images_std
            image = image[channel_order_dict[self.channel_order], :, :]
            return image

        def _data_reader():
            if phase == "predict":
                for image_path in data:
                    image = preprocess(image_path)
                    yield (image, )
            else:
                for image_path, label in data:
                    image = preprocess(image_path)
                    yield (image, label)

        return paddle.batch(_data_reader, batch_size=batch_size)

    def get_train_examples(self):
        return self.dataset.train_examples

    def get_dev_examples(self):
        return self.dataset.dev_examples

    def get_test_examples(self):
        return self.dataset.test_examples
