# coding:utf-8
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
from .base_reader import BaseReader
from ..contrib.ppdet.data.reader import Reader
from ..common import detection_config as dconf

channel_order_dict = {
    "RGB": [0, 1, 2],
    "RBG": [0, 2, 1],
    "GBR": [1, 2, 0],
    "GRB": [1, 0, 2],
    "BGR": [2, 1, 0],
    "BRG": [2, 0, 1]
}


class ImageClassificationReader(BaseReader):
    def __init__(self,
                 image_width,
                 image_height,
                 dataset=None,
                 channel_order="RGB",
                 images_mean=None,
                 images_std=None,
                 data_augmentation=False,
                 random_seed=None):
        super(ImageClassificationReader, self).__init__(dataset, random_seed)
        self.image_width = image_width
        self.image_height = image_height
        self.channel_order = channel_order
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
                       batch_size=1,
                       phase="train",
                       shuffle=False,
                       data=None,
                       return_list=True):
        if phase != 'predict' and not self.dataset:
            raise ValueError("The dataset is none and it's not allowed!")
        if phase == "train":
            shuffle = True
            if hasattr(self.dataset, "train_data"):
                # Compatible with ImageClassificationDataset which has done shuffle
                self.dataset.train_data()
                shuffle = False
            data = self.get_train_examples()
            self.num_examples['train'] = len(data)
        elif phase == "val" or phase == "dev":
            shuffle = False
            if hasattr(self.dataset, "validate_data"):
                # Compatible with ImageClassificationDataset
                self.dataset.validate_data()
                shuffle = False
            data = self.get_dev_examples()
            self.num_examples['dev'] = len(data)
        elif phase == "test":
            shuffle = False
            if hasattr(self.dataset, "test_data"):
                # Compatible with ImageClassificationDataset
                data = self.dataset.test_data()
                shuffle = False
            data = self.get_test_examples()
            self.num_examples['test'] = len(data)
        elif phase == "predict":
            shuffle = False
            data = data

        def preprocess(image_path):
            image = Image.open(image_path)
            image = image_augmentation.image_resize(image, self.image_width,
                                                    self.image_height)
            if self.data_augmentation and phase == "train":
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
            if shuffle:
                np.random.shuffle(data)
            images = []
            labels = []
            if phase == "predict":
                for image_path in data:
                    image = preprocess(image_path)
                    images.append(image.astype('float32'))
                    if len(images) == batch_size:
                        # predictor must receive numpy array not list
                        images = np.array([images]).astype('float32')
                        if return_list:
                            # for DataFeeder
                            yield [images]
                        else:
                            # for DataLoader
                            yield images
                        images = []
                if images:
                    images = np.array([images]).astype('float32')
                    if return_list:
                        yield [images]
                    else:
                        yield images
                    images = []
            else:
                for image_path, label in data:
                    image = preprocess(image_path)
                    images.append(image.astype('float32'))
                    labels.append([np.int64(label)])

                    if len(images) == batch_size:
                        if return_list:
                            yield [[images, labels]]
                        else:
                            yield [images, labels]
                        images = []
                        labels = []
                if images:
                    if return_list:
                        yield [[images, labels]]
                    else:
                        yield [images, labels]
                    images = []
                    labels = []

        return _data_reader


class ObjectDetectionReader(BaseReader):
    def __init__(self,
                 dataset=None,
                 model_type='ssd',
                 worker_num=1,
                 use_process=False,
                 random_seed=None):
        super(ObjectDetectionReader, self).__init__(
            dataset, random_seed=random_seed)
        self.model_type = model_type
        self.worker_num = worker_num
        self.use_process = use_process

    def data_generator(self,
                       batch_size,
                       phase="train",
                       shuffle=False,
                       data=None,
                       return_list=False):
        if phase != 'predict' and not self.dataset:
            raise ValueError("The dataset is none and it's not allowed!")
        drop_last = False
        if phase == "train":
            data_src = self.dataset.train_data(shuffle)
            self.num_examples['train'] = len(self.get_train_examples())
            drop_last = True
        elif phase == "test":
            shuffle = False
            data_src = self.dataset.test_data(shuffle)
            self.num_examples['test'] = len(self.get_test_examples())
        elif phase == "val" or phase == "dev":
            shuffle = False
            data_src = self.dataset.validate_data(shuffle)
            self.num_examples['dev'] = len(self.get_dev_examples())
        else:  # phase == "predict":
            from ..contrib.ppdet.data.source import build_source
            data_config = {"IMAGES": data, "TYPE": "SimpleSource"}
            data_src = build_source(data_config)

        data_cf = {}
        transform_config = {
            'WORKER_CONF': {
                'bufsize': 20,
                'worker_num': self.worker_num,
                'use_process': self.use_process,
                'memsize': '3G'
            },
            'BATCH_SIZE': batch_size,
            'DROP_LAST': drop_last,
            'USE_PADDED_IM_INFO': False,
        }

        phase_trans = {"val": "dev", "test": "dev", "inference": "predict"}
        if phase in phase_trans:
            phase = phase_trans[phase]
        assert phase in ('train', 'dev', 'predict')
        feed_config = dconf.feed_config[self.model_type][phase]
        transform_config.update(feed_config)  # add 'OPS' etc.

        ppdet_mode = 'VAL' if phase != 'train' else 'TRAIN'

        batch_reader = Reader.create(
            ppdet_mode, data_cf, transform_config, my_source=data_src)
        # return itr
        # When call `_batch_reader()`, then return generator(or iterator)
        return batch_reader
