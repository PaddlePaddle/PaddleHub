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
from paddlehub.dataset.base_nlp_dataset import TextClassificationDataset
from paddlehub.common.dir import DATA_HOME

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/tnews.tar.gz"

LABEL_NAME = {
    "100": "news_story",
    "101": "news_culture",
    "102": "news_entertainment",
    "103": "news_sports",
    "104": "news_finance",
    "106": "news_house",
    "107": "news_car",
    "108": "news_edu",
    "109": "news_tech",
    "110": "news_military",
    "112": "news_travel",
    "113": "news_world",
    "114": "stock",
    "115": "news_agriculture",
    "116": "news_game"
}


class TNews(TextClassificationDataset):
    """
    TNews is the chinese news classification dataset on Jinri Toutiao App.
    """

    def __init__(self, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(DATA_HOME, "tnews")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        label_list = [
            '100', '101', '102', '103', '104', '106', '107', '108', '109',
            '110', '112', '113', '114', '115', '116'
        ]
        super(TNews, self).__init__(
            base_path=base_path,
            train_file="toutiao_category_train.txt",
            dev_file="toutiao_category_dev.txt",
            test_file="toutiao_category_test.txt",
            label_file=None,
            label_list=label_list,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)

    def get_label_name(self, id):
        return LABEL_NAME[id]

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as file:
            examples = []
            for line in file:
                data = line.strip().split("_!_")
                example = InputExample(
                    guid=data[0], label=data[1], text_a=data[3])
                examples.append(example)

            return examples


if __name__ == "__main__":
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='vocab.txt')
    ds = TNews(tokenizer=tokenizer, max_seq_len=10)
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
