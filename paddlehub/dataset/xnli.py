# default_downloader.download_file_and_uncompress(
#                 url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
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

import io
import os
import csv

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import TextClassificationDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/XNLI-lan.tar.gz"


class XNLI(TextClassificationDataset):
    """
    Please refer to
    https://arxiv.org/pdf/1809.05053.pdf
    for more information
    """

    def __init__(self, language='zh', tokenizer=None, max_seq_len=None):
        if language not in [
                "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw",
                "th", "tr", "ur", "vi", "zh"
        ]:
            raise Exception(
                "%s is not in XNLI. Please confirm the language" % language)
        self.language = language
        dataset_dir = os.path.join(DATA_HOME, "XNLI-lan")
        dataset_dir = self._download_dataset(dataset_dir, url=_DATA_URL)
        base_path = os.path.join(dataset_dir, language)
        super(XNLI, self).__init__(
            base_path=base_path,
            train_file="%s_train.tsv" % language,
            dev_file="%s_dev.tsv" % language,
            test_file="%s_test.tsv" % language,
            label_file=None,
            label_list=["neutral", "contradiction", "entailment"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[2], text_a=line[0], text_b=line[1])
                seq_id += 1
                examples.append(example)

            return examples


if __name__ == "__main__":
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='vocab.txt')

    ds = XNLI(tokenizer=tokenizer, max_seq_len=20)
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
