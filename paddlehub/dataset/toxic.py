#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import codecs
import os
import pandas as pd
from numpy import nan

from paddlehub.dataset import InputExample, HubDataset
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/toxic.tar.gz"


class Toxic(HubDataset):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)
    """

    def __init__(self):
        self.dataset_dir = os.path.join(DATA_HOME, "toxic")
        if not os.path.exists(self.dataset_dir):
            ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.csv")
        self.train_examples = self._read_csv(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.csv")
        self.dev_examples = self._read_csv(self.dev_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.csv")
        self.test_examples = self._read_csv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
            'identity_hate'
        ]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_csv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = pd.read_csv(input_file, encoding="UTF-8")
        examples = []
        for index, row in data.iterrows():
            guid = row["id"]
            text = row["comment_text"]
            labels = [int(value) for value in row[2:]]
            example = InputExample(guid=guid, label=labels, text_a=text)
            examples.append(example)

        return examples


if __name__ == "__main__":
    ds = Toxic()
    for e in ds.get_train_examples():
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
