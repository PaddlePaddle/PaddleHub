# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
from typing import Tuple, Callable

import paddle
import numpy as np
from PIL import Image


class SegDataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (Callable): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
        edge (bool, optional): Whether to compute edge while training. Default: False

    """

    def __init__(self,
                 transforms: Callable,
                 dataset_root: str,
                 num_classes: int,
                 mode: str = 'train',
                 train_path: str = None,
                 val_path: str = None,
                 test_path: str = None,
                 separator: str = ' ',
                 ignore_index: int = 255,
                 edge: bool = False):
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge

        if mode.lower() not in ['train', 'val', 'test']:
            raise ValueError("mode should be 'train', 'val' or 'test', but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(self.dataset_root))

        if mode == 'train':
            if train_path is None:
                raise ValueError('When `mode` is "train", `train_path` is necessary, but it is None.')
            elif not os.path.exists(train_path):
                raise FileNotFoundError('`train_path` is not found: {}'.format(train_path))
            else:
                file_path = train_path
        elif mode == 'val':
            if val_path is None:
                raise ValueError('When `mode` is "val", `val_path` is necessary, but it is None.')
            elif not os.path.exists(val_path):
                raise FileNotFoundError('`val_path` is not found: {}'.format(val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError('When `mode` is "test", `test_path` is necessary, but it is None.')
            elif not os.path.exists(test_path):
                raise FileNotFoundError('`test_path` is not found: {}'.format(test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise ValueError("File list format incorrect! In training or evaluation task it should be"
                                         " image_name{}label_name\\n".format(separator))
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray]:
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            return im, label

    def __len__(self) -> int:
        return len(self.file_list)
