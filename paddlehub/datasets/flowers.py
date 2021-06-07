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
from typing import Callable, Tuple

import paddle
import numpy as np

import paddlehub.env as hubenv
from paddlehub.utils.download import download_data


@download_data(url='https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz')
class Flowers(paddle.io.Dataset):
    def __init__(self, transforms: Callable, mode: str = 'train'):
        self.mode = mode
        self.transforms = transforms
        self.num_classes = 5

        if self.mode == 'train':
            self.file = 'train_list.txt'
        elif self.mode == 'test':
            self.file = 'test_list.txt'
        else:
            self.file = 'validate_list.txt'
        self.file = os.path.join(hubenv.DATA_HOME, 'flower_photos', self.file)

        with open(self.file, 'r') as file:
            self.data = file.read().split('\n')

    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        img_path, grt = self.data[idx].split(' ')
        img_path = os.path.join(hubenv.DATA_HOME, 'flower_photos', img_path)
        im = self.transforms(img_path)
        return im, int(grt)

    def __len__(self):
        return len(self.data)
