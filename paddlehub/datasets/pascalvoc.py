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
from paddlehub.env import DATA_HOME
from pycocotools.coco import COCO

from paddlehub.process.transforms import DetectCatagory, ParseImages


class DetectionData(paddle.io.Dataset):
    """
    Dataset for image detection.

    Args:
       transform(callmethod) : The method of preprocess images.
       mode(str): The mode for preparing dataset.

    Returns:
        DataSet: An iterable object for data iterating
    """
    def __init__(self, transform: Callable, size: int = 416, mode: str = 'train'):
        self.mode = mode
        self.transform = transform
        self.size = size

        if self.mode == 'train':
            train_file_list = 'annotations/instances_train2017.json'
            train_data_dir = 'train2017'
            self.train_file_list = os.path.join(DATA_HOME, 'voc', train_file_list)
            self.train_data_dir = os.path.join(DATA_HOME, 'voc', train_data_dir)
            self.COCO = COCO(self.train_file_list)
            self.img_dir = self.train_data_dir

        elif self.mode == 'test':
            val_file_list = 'annotations/instances_val2017.json'
            val_data_dir = 'val2017'
            self.val_file_list = os.path.join(DATA_HOME, 'voc', val_file_list)
            self.val_data_dir = os.path.join(DATA_HOME, 'voc', val_data_dir)
            self.COCO = COCO(self.val_file_list)
            self.img_dir = self.val_data_dir

        parse_dataset_catagory = DetectCatagory(self.COCO, self.img_dir)
        self.label_names, self.label_ids, self.category_to_id_map = parse_dataset_catagory()
        parse_images = ParseImages(self.COCO, self.mode, self.img_dir, self.category_to_id_map)
        self.data = parse_images()

    def __getitem__(self, idx: int):
        if self.mode == "train":
            img = self.data[idx]
            out_img, gt_boxes, gt_labels, gt_scores = self.transform(img, 416)
            return out_img, gt_boxes, gt_labels, gt_scores
        elif self.mode == "test":
            img = self.data[idx]
            out_img, id, (h, w) = self.transform(img)
            return out_img, id, (h, w)

    def __len__(self):
        return len(self.data)
