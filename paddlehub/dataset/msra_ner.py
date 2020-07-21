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
import codecs
import csv

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import SeqLabelingDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/msra_ner.tar.gz"


class MSRA_NER(SeqLabelingDataset):
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """

    def __init__(self, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(DATA_HOME, "msra_ner")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        super(MSRA_NER, self).__init__(
            base_path=base_path,
            train_file="train.tsv",
            dev_file="dev.tsv",
            test_file="test.tsv",
            label_file=None,
            label_list=[
                "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"
            ],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[1], text_a=line[0])
                seq_id += 1
                examples.append(example)

            return examples


if __name__ == "__main__":
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='vocab.txt')
    ds = MSRA_NER(tokenizer=tokenizer, max_seq_len=30)
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
