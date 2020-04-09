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

from paddlehub.dataset import InputExample
from paddlehub.common.logger import logger
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/glue_data.tar.gz"


class GLUE(BaseNLPDataset):
    """
    Please refer to
    https://gluebenchmark.com
    for more information
    """

    def __init__(self, sub_dataset='SST-2'):
        # sub_dataset : CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B
        if sub_dataset not in [
                'CoLA', 'MNLI', 'MNLI_m', 'MNLI_mm', 'MRPC', 'QNLI', 'QQP',
                'RTE', 'SST-2', 'STS-B'
        ]:
            raise Exception(
                "%s is not in GLUE benchmark. Please confirm the data set" %
                sub_dataset)

        mismatch = False
        if sub_dataset == 'MNLI_mm':
            sub_dataset = 'MNLI'
            mismatch = True
        elif sub_dataset == 'MNLI_m':
            sub_dataset = 'MNLI'
        self.sub_dataset = sub_dataset

        # test.tsv has not label,so it is a predict file
        dev_file = "dev.tsv"
        predict_file = "test.tsv"
        if sub_dataset == 'MNLI' and not mismatch:
            dev_file = 'dev_matched.tsv'
            predict_file = "test_matched.tsv"
        elif sub_dataset == 'MNLI' and mismatch:
            dev_file = 'dev_mismatched.tsv'
            predict_file = "test_mismatched.tsv"

        dataset_dir = os.path.join(DATA_HOME, "glue_data")
        dataset_dir = self._download_dataset(dataset_dir, url=_DATA_URL)
        base_path = os.path.join(dataset_dir, self.sub_dataset)

        label_list = None
        if sub_dataset in ['MRPC', 'QQP', 'SST-2', 'CoLA']:
            label_list = ["0", "1"]
        elif sub_dataset in ['QNLI', 'RTE']:
            label_list = ['not_entailment', 'entailment']
        elif sub_dataset in ['MNLI']:
            label_list = ["neutral", "contradiction", "entailment"]
        elif sub_dataset in ['STS-B']:
            label_list = None

        super(GLUE, self).__init__(
            base_path=base_path,
            train_file="train.tsv",
            dev_file=dev_file,
            predict_file=predict_file,
            label_file=None,
            label_list=label_list,
        )

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            examples = []
            seq_id = 0
            if self.sub_dataset != 'CoLA' or phase == "predict":
                header = next(reader)  # skip header
            if self.sub_dataset in [
                    'MRPC',
            ]:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, -2, -1]
                else:
                    label_index, text_a_index, text_b_index = [0, -2, -1]
            elif self.sub_dataset in [
                    'QNLI',
            ]:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, 1, 2]
                else:
                    label_index, text_a_index, text_b_index = [3, 1, 2]
            elif self.sub_dataset in [
                    'QQP',
            ]:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, 1, 2]
                else:
                    label_index, text_a_index, text_b_index = [5, 3, 4]
            elif self.sub_dataset in [
                    'RTE',
            ]:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, 1, 2]
                else:
                    label_index, text_a_index, text_b_index = [3, 1, 2]
            elif self.sub_dataset in [
                    'SST-2',
            ]:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, 1, None]
                else:
                    label_index, text_a_index, text_b_index = [1, 0, None]
            elif self.sub_dataset in [
                    'MNLI',
            ]:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, 8, 9]
                else:
                    label_index, text_a_index, text_b_index = [-1, 8, 9]
            elif self.sub_dataset in ['CoLA']:
                if phase == "predict":
                    label_index, text_a_index, text_b_index = [None, 1, None]
                else:
                    label_index, text_a_index, text_b_index = [1, 3, None]
            elif self.sub_dataset in ['STS-B']:
                if phase == "predict":
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
                    logger.info("[Discard Incorrect Data] " + "\t".join(line))
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
