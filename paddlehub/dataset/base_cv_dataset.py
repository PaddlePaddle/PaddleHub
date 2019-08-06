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

import numpy as np

import paddlehub as hub
from paddlehub.common.downloader import default_downloader


class ImageClassificationDataset(object):
    def __init__(self):
        self.base_path = None
        self.train_list_file = None
        self.test_list_file = None
        self.validate_list_file = None
        self.label_list_file = None
        self.num_labels = 0
        self.label_list = []

        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

    def _download_dataset(self, dataset_path, url):
        if not os.path.exists(dataset_path):
            result, tips, dataset_path = default_downloader.download_file_and_uncompress(
                url=url,
                save_path=hub.common.dir.DATA_HOME,
                print_progress=True,
                replace=True)
            if not result:
                print(tips)
                exit()
        return dataset_path

    def _parse_data(self, data_path, shuffle=False, phase=None):
        def _base_reader():
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
                            image_path = os.path.join(self.base_path,
                                                      image_path)
                    label = items[-1]
                    data.append((image_path, items[-1]))

            if phase == 'train':
                self.train_examples = data
            elif phase == 'dev':
                self.dev_examples = data
            elif phase == 'test':
                self.test_examples = data

            if shuffle:
                np.random.shuffle(data)

            for item in data:
                yield item

        return _base_reader()

    def label_dict(self):
        if not self.label_list:
            with open(os.path.join(self.base_path, self.label_list_file),
                      "r") as file:
                self.label_list = file.read().split("\n")
        return {index: key for index, key in enumerate(self.label_list)}

    def train_data(self, shuffle=True):
        train_data_path = os.path.join(self.base_path, self.train_list_file)
        return self._parse_data(train_data_path, shuffle, phase='train')

    def test_data(self, shuffle=False):
        test_data_path = os.path.join(self.base_path, self.test_list_file)
        return self._parse_data(test_data_path, shuffle, phase='dev')

    def validate_data(self, shuffle=False):
        validate_data_path = os.path.join(self.base_path,
                                          self.validate_list_file)
        return self._parse_data(validate_data_path, shuffle, phase='test')

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples
