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

from collections import namedtuple
import io
import os
import csv

from paddlehub.dataset import InputExample, HubDataset
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/inews.tar.gz"


class INews(HubDataset):
    """
    INews is a sentiment analysis dataset for Internet News
    """

    def __init__(self):
        self.dataset_dir = os.path.join(DATA_HOME, "inews")
        if not os.path.exists(self.dataset_dir):
            ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.txt")
        self.train_examples = self._read_file(self.train_file, is_training=True)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.txt")
        self.dev_examples = self._read_file(self.dev_file, is_training=False)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.txt")
        self.test_examples = self._read_file(self.test_file, is_training=False)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return ["0", "1", "2"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_file(self, input_file, is_training):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as file:
            examples = []
            for (i, line) in enumerate(file):
                if i == 0 and is_training:
                    continue
                data = line.strip().split("_!_")
                example = InputExample(
                    guid=i, label=data[0], text_a=data[2], text_b=data[3])
                examples.append(example)
            return examples


if __name__ == "__main__":
    ds = INews()
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
