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
from typing import Callable

import paddle
import numpy as np

import paddlehub.env as hubenv
from paddlehub.vision.utils import get_img_file
from paddlehub.utils.download import download_data


@download_data(url='https://paddlehub.bj.bcebos.com/dygraph/datasets/minicoco.tar.gz')
class MiniCOCO(paddle.io.Dataset):
    """
    Dataset for Style transfer. The dataset contains 2001 images for training set and 200 images for testing set.
    They are derived form COCO2014. Meanwhile, it contains 21 different style pictures in file "21styles".

    Args:
       transform(callmethod) : The method of preprocess images.
       mode(str): The mode for preparing dataset.

    Returns:
        DataSet: An iterable object for data iterating
    """

    def __init__(self, transform: Callable, mode: str = 'train'):
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            self.file = 'train'
        elif self.mode == 'test':
            self.file = 'test'
        self.file = os.path.join(hubenv.DATA_HOME, 'minicoco', self.file)
        self.style_file = os.path.join(hubenv.DATA_HOME, 'minicoco', '21styles')
        self.data = get_img_file(self.file)
        self.style = get_img_file(self.style_file)

    def __getitem__(self, idx: int) -> np.ndarray:

        img_path = self.data[idx]
        im = self.transform(img_path)
        im = im.astype('float32')
        style_idx = idx % len(self.style)
        style_path = self.style[style_idx]
        style = self.transform(style_path)
        style = style.astype('float32')
        return im, style

    def __len__(self):
        return len(self.data)
