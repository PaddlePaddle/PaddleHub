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
import os
import csv

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/inews.tar.gz"


class INews(BaseNLPDataset):
    """
    INews is a sentiment analysis dataset for Internet News
    """

    def __init__(self):
        dataset_dir = os.path.join(DATA_HOME, "inews")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        super(INews, self).__init__(
            base_path=base_path,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            label_file=None,
            label_list=["0", "1", "2"],
        )

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as file:
            examples = []
            for (i, line) in enumerate(file):
                if i == 0 and phase == 'train':
                    continue
                data = line.strip().split("_!_")
                example = InputExample(
                    guid=i, label=data[0], text_a=data[2], text_b=data[3])
                examples.append(example)
            return examples


if __name__ == "__main__":
    ds = INews()
    print("first 10 dev")
    for e in ds.get_dev_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print("first 10 train")
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print("first 10 test")
    for e in ds.get_test_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print(ds)
