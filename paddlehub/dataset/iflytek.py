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

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import TextClassificationDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/iflytek.tar.gz"


class IFLYTEK(TextClassificationDataset):
    def __init__(self, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(DATA_HOME, "iflytek")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        super(IFLYTEK, self).__init__(
            base_path=base_path,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            label_file=None,
            label_list=[str(i) for i in range(119)],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as file:
            examples = []
            for (i, line) in enumerate(file):
                data = line.strip().split("_!_")
                try:
                    example = InputExample(
                        guid=i, label=str(data[0]), text_a=data[1], text_b=None)
                    examples.append(example)
                except:
                    pass
            return examples


if __name__ == "__main__":
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='vocab.txt')
    ds = IFLYTEK(tokenizer=tokenizer, max_seq_len=10)
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
    print("first 10 dev records")
    for e in ds.get_dev_records()[:10]:
        print(e)
