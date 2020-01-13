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

import os
import pandas as pd

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/toxic.tar.gz"


class Toxic(BaseNLPDataset):
    """
    The kaggle Toxic dataset:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(self):
        dataset_dir = os.path.join(DATA_HOME, "toxic")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        label_list = [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
            'identity_hate'
        ]
        super(Toxic, self).__init__(
            base_path=base_path,
            train_file="train.csv",
            dev_file="dev.csv",
            test_file="test.csv",
            label_file=None,
            label_list=label_list,
        )

    def _read_file(self, input_file, phase=None):
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
