# coding:utf-8
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

import io

from paddlehub.dataset import InputExample, HubDataset


class Base_NLP_Dataset(HubDataset):
    def __init__(self,
                 base_path,
                 train_file=None,
                 dev_file=None,
                 test_file=None,
                 predict_file=None,
                 label_file=None,
                 label_list=None):
        super(Base_NLP_Dataset, self).__init__(
            base_path=base_path,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            predict_file=predict_file,
            label_file=label_file,
            label_list=label_list)

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as file:
            examples = []
            for (i, line) in enumerate(file):
                data = line.strip().split("\t")
                example = InputExample(
                    guid=i, label=data[2], text_a=data[0], text_b=data[1])
                examples.append(example)
            return examples
