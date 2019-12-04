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

from paddlehub.dataset import HubDataset


class ImageClassificationDataset(HubDataset):
    def __init__(self,
                 base_path,
                 train_list_file=None,
                 validate_list_file=None,
                 test_list_file=None,
                 label_list_file=None,
                 label_list=None):
        super(ImageClassificationDataset, self).__init__(
            base_path=base_path,
            train_file=train_list_file,
            dev_file=validate_list_file,
            test_file=test_list_file,
            label_file=label_list_file,
            label_list=label_list)

    def _read_file(self, data_path, phase=None):
        data = []
        with open(data_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip()
                items = line.split(" ")
                if len(items) > 2:
                    image_path = " ".join(items[0:-1])
                else:
                    image_path = items[0]
                if not os.path.isabs(image_path):
                    if self.base_path is not None:
                        image_path = os.path.join(self.base_path, image_path)
                label = items[-1]
                data.append((image_path, label))
        return data
