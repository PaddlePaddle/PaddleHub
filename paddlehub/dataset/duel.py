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
from paddlehub.dataset.base_nlp_dataset import TextMatchingDataset


class DuEL(TextMatchingDataset):
    """
    DuEL is  a collection of question pairs from CCKS 2020 competition, which is a pair-wise text macthing dataset.
    More information, please refer to https://github.com/PaddlePaddle/Research/tree/master/KG/DuEL_Baseline.
    """

    def __init__(self, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(DATA_HOME, "duel")
        base_path = self._download_dataset(
            dataset_dir,
            url="https://bj.bcebos.com/paddlehub/dataset/duel.tar.gz")
        super(DuEL, self).__init__(
            is_pair_wise=True,
            base_path=base_path,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file=None,
            predict_file="test.txt",
            train_file_with_header=True,
            dev_file_with_header=True,
            predict_file_with_header=True,
            label_list=["0", "1"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)


if __name__ == "__main__":
    from paddlehub.tokenizer.tokenizer import CustomTokenizer
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(
        vocab_file='/mnt/zhangxuefei/.paddlehub/modules/ernie/assets/vocab.txt',
        tokenize_chinese_chars=False)
    ds = DuEL(tokenizer=tokenizer, max_seq_len=60)
    print("first 10 train examples")
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.text_c,
                                      e.label))
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
