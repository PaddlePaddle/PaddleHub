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

import os

from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import TextClassificationDataset


class BQ(TextClassificationDataset):
    """
    The Bank Question (BQ) corpus, a Chinese corpus for sentence semantic equivalence identification (SSEI),
    contains 120,000 question pairs from 1-year online bank custom service logs.
    """

    def __init__(self, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(DATA_HOME, "bq")
        base_path = self._download_dataset(
            dataset_dir,
            url="https://bj.bcebos.com/paddlehub-dataset/bq.tar.gz")
        super(BQ, self).__init__(
            base_path=base_path,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            label_file=None,
            label_list=["0", "1"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)


if __name__ == "__main__":
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    ds = BQ(tokenizer=BertTokenizer(vocab_file='vocab.txt'), max_seq_len=10)
    print("first 10 dev examples")
    for e in ds.get_dev_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print("first 10 dev records")
    for e in ds.get_dev_records()[:10]:
        print(e)
