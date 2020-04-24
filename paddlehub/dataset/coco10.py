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

import os

import paddlehub as hub
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset


class Coco10(ObjectDetectionDataset):
    def __init__(self, model_type='ssd'):
        dataset_path = os.path.join(hub.common.dir.DATA_HOME, "coco_10")
        # self.base_path = self._download_dataset(
        #     dataset_path=dataset_path,
        #     url="https://bj.bcebos.com/paddlehub-dataset/dog-cat.tar.gz")
        base_path = dataset_path
        train_image_dir = 'val'
        train_list_file = 'annotations/val.json'
        validate_image_dir = 'val'
        validate_list_file = 'annotations/val.json'
        test_image_dir = 'val'
        test_list_file = 'annotations/val.json'
        super(Coco10, self).__init__(
            base_path, train_image_dir, train_list_file, validate_image_dir,
            validate_list_file, test_image_dir, test_list_file, model_type)
