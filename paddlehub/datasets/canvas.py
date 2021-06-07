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


@download_data(url='https://paddlehub.bj.bcebos.com/dygraph/datasets/canvas.tar.gz')
class Canvas(paddle.io.Dataset):
    """
    Dataset for colorization. It contains 1193 and 400 pictures for Monet and Vango paintings style, respectively.
    We collected data from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/.

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

        self.file = os.path.join(hubenv.DATA_HOME, 'canvas', self.file)
        self.data = get_img_file(self.file)

    def __getitem__(self, idx: int) -> np.ndarray:
        img_path = self.data[idx]
        im = self.transform(img_path)
        return im

    def __len__(self):
        return len(self.data)
