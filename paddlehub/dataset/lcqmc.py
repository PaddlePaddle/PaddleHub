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

import codecs
import os
import csv

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import TextMatchingDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz"


class LCQMC(TextMatchingDataset):
    def __init__(self, is_pair_wise=False, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(DATA_HOME, "lcqmc")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        super(LCQMC, self).__init__(
            is_pair_wise=is_pair_wise,
            base_path=base_path,
            train_file="train.tsv",
            dev_file="dev.tsv",
            test_file="test.tsv",
            label_file=None,
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            label_list=["0", "1"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)


if __name__ == "__main__":
    from paddlehub.tokenizer.tokenizer import CustomTokenizer
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = CustomTokenizer(
        vocab_file=
        '/mnt/zhangxuefei/.paddlehub/modules/senta_bow/assets/vocab.txt',
        tokenize_chinese_chars=True)
    ds = LCQMC(tokenizer=tokenizer, max_seq_len=60)
    print("first 10 train examples")
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print("first 10 train records")
    for e in ds.get_train_records()[:10]:
        print(e)
    print("first 10 test records")
    for e in ds.get_test_records()[:10]:
        print(e)
    print("first 10 predict records")
    for e in ds.get_predict_records()[:10]:
        print(e)

    print(ds.get_feed_list("train"))
    print(ds.get_feed_list("dev"))
    print(ds.get_feed_list("test"))
    print(ds.get_feed_list("predict"))
