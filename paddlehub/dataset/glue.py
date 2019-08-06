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
import csv
import io

from paddlehub.dataset import InputExample, HubDataset
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/glue_data.tar.gz"


class GLUE(HubDataset):
    """
    Please refer to
    https://gluebenchmark.com
    for more information
    """

    def __init__(self, sub_dataset='SST-2'):
        # sub_dataset : CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B
        if sub_dataset not in [
                'CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B'
        ]:
            raise Exception(
                sub_dataset +
                " is not in GLUE benchmark. Please confirm the data set")
        self.sub_dataset = sub_dataset
        self.dataset_dir = os.path.join(DATA_HOME, "glue_data")

        if not os.path.exists(self.dataset_dir):
            ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_dev_examples()
        self._load_test_examples()
        self._load_predict_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, self.sub_dataset,
                                       "train.tsv")
        self.train_examples = self._read_tsv(self.train_file)

    def _load_dev_examples(self):
        if self.sub_dataset == 'MNLI':
            self.dev_file = os.path.join(self.dataset_dir, self.sub_dataset,
                                         "dev_matched.tsv")
        else:
            self.dev_file = os.path.join(self.dataset_dir, self.sub_dataset,
                                         "dev.tsv")
        self.dev_examples = self._read_tsv(self.dev_file)

    def _load_test_examples(self):
        self.test_examples = []

    def _load_predict_examples(self):
        if self.sub_dataset == 'MNLI':
            self.predict_file = os.path.join(self.dataset_dir, self.sub_dataset,
                                             "test_matched.tsv")
        else:
            self.predict_file = os.path.join(self.dataset_dir, self.sub_dataset,
                                             "test.tsv")
        self.predict_examples = self._read_tsv(self.predict_file, wo_label=True)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_predict_examples(self):
        return self.predict_examples

    def get_labels(self):
        """See base class."""
        if self.sub_dataset in ['MRPC', 'QQP', 'SST-2', 'CoLA']:
            return ["0", "1"]
        elif self.sub_dataset in ['QNLI', 'RTE']:
            return ['not_entailment', 'entailment']
        elif self.sub_dataset in ['MNLI']:
            return ["neutral", "contradiction", "entailment"]
        elif self.sub_dataset in ['STS-B']:
            return Exception("No category labels for regreesion tasks")

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_tsv(self, input_file, quotechar=None, wo_label=False):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = []
            seq_id = 0
            if self.sub_dataset != 'CoLA' or wo_label:
                header = next(reader)  # skip header
            if self.sub_dataset in [
                    'MRPC',
            ]:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, -2, -1]
                else:
                    label_index, text_a_index, text_b_index = [0, -2, -1]
            elif self.sub_dataset in [
                    'QNLI',
            ]:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, 1, 2]
                else:
                    label_index, text_a_index, text_b_index = [3, 1, 2]
            elif self.sub_dataset in [
                    'QQP',
            ]:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, 1, 2]
                else:
                    label_index, text_a_index, text_b_index = [5, 3, 4]
            elif self.sub_dataset in [
                    'RTE',
            ]:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, 1, 2]
                else:
                    label_index, text_a_index, text_b_index = [3, 1, 2]
            elif self.sub_dataset in [
                    'SST-2',
            ]:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, 1, None]
                else:
                    label_index, text_a_index, text_b_index = [1, 0, None]
            elif self.sub_dataset in [
                    'MNLI',
            ]:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, 8, 9]
                else:
                    label_index, text_a_index, text_b_index = [-1, 8, 9]
            elif self.sub_dataset in ['CoLA']:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, 1, None]
                else:
                    label_index, text_a_index, text_b_index = [1, 3, None]
            elif self.sub_dataset in ['STS-B']:
                if wo_label:
                    label_index, text_a_index, text_b_index = [None, -2, -1]
                else:
                    label_index, text_a_index, text_b_index = [-1, -3, -2]

            for line in reader:
                try:
                    example = InputExample(
                        guid=seq_id,
                        text_a=line[text_a_index],
                        text_b=line[text_b_index]
                        if text_b_index is not None else None,
                        label=line[label_index]
                        if label_index is not None else None)
                    seq_id += 1
                    examples.append(example)
                except:
                    print("[Discard Incorrect Data] " + "\t".join(line))
            return examples


if __name__ == "__main__":
    for sub_dataset in [
            'CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B'
    ]:
        print(sub_dataset)
        ds = GLUE(sub_dataset=sub_dataset)
        for e in ds.get_train_examples()[:2]:
            print(e)
        print()
        for e in ds.get_dev_examples()[:2]:
            print(e)
        print()
        for e in ds.get_test_examples()[:2]:
            print(e)
        print()
        for e in ds.get_predict_examples()[:2]:
            print(e)
        print()
