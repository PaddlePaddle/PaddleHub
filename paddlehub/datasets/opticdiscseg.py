# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
from paddlehub.datasets.base_seg_dataset import SegDataset


@download_data(url='https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip')
class OpticDiscSeg(SegDataset):
    """
    OpticDiscSeg dataset is extraced from iChallenge-AMD
    (https://ai.baidu.com/broad/subordinate?dataset=amd).

    Args:
        transforms (Callable): Transforms for image.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """

    def __init__(self, transforms: Callable = None, mode: str = 'train'):
        self.transforms = transforms
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = 2
        self.ignore_index = 255

        if mode not in ['train', 'val', 'test']:
            raise ValueError("`mode` should be 'train', 'val' or 'test', but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if mode == 'train':
            file_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', 'train_list.txt')
        elif mode == 'test':
            file_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', 'test_list.txt')
        else:
            file_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', 'val_list.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception("File list format incorrect! It should be" " image_name label_name\\n")
                    image_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', items[0])
                    grt_path = os.path.join(hubenv.DATA_HOME, 'optic_disc_seg', items[1])
                self.file_list.append([image_path, grt_path])
