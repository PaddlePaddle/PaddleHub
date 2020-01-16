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
        super(Coco10, self).__init__(model_type)
        dataset_path = os.path.join(hub.common.dir.DATA_HOME, "coco_10")
        # self.base_path = self._download_dataset(
        #     dataset_path=dataset_path,
        #     url="https://bj.bcebos.com/paddlehub-dataset/dog-cat.tar.gz")
        self.base_path = dataset_path
        self.train_image_dir = 'val'
        self.train_list_file = 'annotations/val.json'
        self.validate_image_dir = 'val'
        self.validate_list_file = 'annotations/val.json'
        self.test_image_dir = 'val'
        self.test_list_file = 'annotations/val.json'
