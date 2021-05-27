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
import copy
from typing import Callable

import paddle
import numpy as np
from paddlehub.env import DATA_HOME
from pycocotools.coco import COCO


class DetectCatagory:
    """Load label name, id and map from detection dataset.

    Args:
        attrbox(Callable): Method to get detection attributes of images.
        data_dir(str): Image dataset path.

    Returns:
        label_names(List(str)): The dataset label names.
        label_ids(List(int)): The dataset label ids.
        category_to_id_map(dict): Mapping relations of category and id for images.
    """

    def __init__(self, attrbox: Callable, data_dir: str):
        self.attrbox = attrbox
        self.img_dir = data_dir

    def __call__(self):
        self.categories = self.attrbox.loadCats(self.attrbox.getCatIds())
        self.num_category = len(self.categories)
        label_names = []
        label_ids = []
        for category in self.categories:
            label_names.append(category['name'])
            label_ids.append(int(category['id']))
        category_to_id_map = {v: i for i, v in enumerate(label_ids)}
        return label_names, label_ids, category_to_id_map


class ParseImages:
    """Prepare images for detection.

    Args:
        attrbox(Callable): Method to get detection attributes of images.
        data_dir(str): Image dataset path.
        category_to_id_map(dict): Mapping relations of category and id for images.

    Returns:
        imgs(dict): The input for detection model, it is a dict.
    """

    def __init__(self, attrbox: Callable, data_dir: str, category_to_id_map: dict):
        self.attrbox = attrbox
        self.img_dir = data_dir
        self.category_to_id_map = category_to_id_map
        self.parse_gt_annotations = GTAnotations(self.attrbox, self.category_to_id_map)

    def __call__(self):
        image_ids = self.attrbox.getImgIds()
        image_ids.sort()
        imgs = copy.deepcopy(self.attrbox.loadImgs(image_ids))

        for img in imgs:
            img['image'] = os.path.join(self.img_dir, img['file_name'])
            assert os.path.exists(img['image']), "image {} not found.".format(img['image'])
            box_num = 50
            img['gt_boxes'] = np.zeros((box_num, 4), dtype=np.float32)
            img['gt_labels'] = np.zeros((box_num), dtype=np.int32)
            img = self.parse_gt_annotations(img)
        return imgs


class GTAnotations:
    """Set gt boxes and gt labels for train.

    Args:
        attrbox(Callable): Method for get detection attributes for images.
        category_to_id_map(dict): Mapping relations of category and id for images.
        img(dict): Input for detection model.

    Returns:
        img(dict): Set specific value on the attributes of 'gt boxes' and 'gt labels' for input.
    """

    def __init__(self, attrbox: Callable, category_to_id_map: dict):
        self.attrbox = attrbox
        self.category_to_id_map = category_to_id_map

    def box_to_center_relative(self, box: list, img_height: int, img_width: int) -> np.ndarray:
        """
            Convert COCO annotations box with format [x1, y1, w, h] to
            center mode [center_x, center_y, w, h] and divide image width
            and height to get relative value in range[0, 1]
        """
        assert len(box) == 4, "box should be a len(4) list or tuple"
        x, y, w, h = box

        x1 = max(x, 0)
        x2 = min(x + w - 1, img_width - 1)
        y1 = max(y, 0)
        y2 = min(y + h - 1, img_height - 1)

        x = (x1 + x2) / 2 / img_width
        y = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        return np.array([x, y, w, h])

    def __call__(self, img: dict):
        img_height = img['height']
        img_width = img['width']
        anno = self.attrbox.loadAnns(self.attrbox.getAnnIds(imgIds=img['id'], iscrowd=None))
        gt_index = 0

        for target in anno:
            if target['area'] < -1:
                continue
            if 'ignore' in target and target['ignore']:
                continue
            box = self.box_to_center_relative(target['bbox'], img_height, img_width)

            if box[2] <= 0 and box[3] <= 0:
                continue
            img['gt_boxes'][gt_index] = box
            img['gt_labels'][gt_index] = \
                self.category_to_id_map[target['category_id']]
            gt_index += 1
            if gt_index >= 50:
                break
        return img


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
        parse_images = ParseImages(self.COCO, self.img_dir, self.category_to_id_map)
        self.data = parse_images()

    def __getitem__(self, idx: int):
        img = self.data[idx]
        im, data = self.transform(img)
        out_img, gt_boxes, gt_labels, gt_scores = im, data['gt_boxes'], data['gt_labels'], data['gt_scores']
        return out_img, gt_boxes, gt_labels, gt_scores

    def __len__(self):
        return len(self.data)
