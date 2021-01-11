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
from PIL import Image

import paddlehub.env as hubenv
from paddlehub.utils.download import download_data


@download_data(url='https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip')
class OpticDiscSeg(paddle.io.Dataset):
    """
    Dataset for image segmentation.

    Args:
       transform(callmethod) : The method of preprocess images.
       mode(str): The mode for preparing dataset.

    Returns:
        DataSet: An iterable object for data iterating
    """

    def __init__(self,
                 transform: Callable,
                 mode: str = "train"):
        self.transforms = transform
        self.mode = mode
        self.num_classes = 2

        if self.mode == 'train':
            self.file = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', 'train_list.txt')
        elif self.mode == 'test':
            self.file = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', 'test_list.txt')
        else:
            self.file = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', 'val_list.txt')
            
        self.data=[]
        with open(self.file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line !='':
                    self.data.append(line)

    def __getitem__(self, idx):
        items = self.data[idx].split(' ')
        image_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', items[0])
        grt_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', items[1])
        im, label = self.transforms(im=image_path, label=grt_path)
        return im, label

    def __len__(self):
        return len(self.data)