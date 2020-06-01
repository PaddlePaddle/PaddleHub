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
import csv
import numpy as np

from paddlehub.dataset import InputExample, BaseDataset
from paddlehub.common.logger import logger


class BaseNLPDataset(BaseDataset):
    def __init__(self,
                 base_path,
                 train_file=None,
                 dev_file=None,
                 test_file=None,
                 predict_file=None,
                 label_file=None,
                 label_list=None,
                 train_file_with_header=False,
                 dev_file_with_header=False,
                 test_file_with_header=False,
                 predict_file_with_header=False,
                 tokenizer=None,
                 max_seq_len=None):
        super(BaseNLPDataset, self).__init__(
            base_path=base_path,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            predict_file=predict_file,
            label_file=label_file,
            label_list=label_list,
            train_file_with_header=train_file_with_header,
            dev_file_with_header=dev_file_with_header,
            test_file_with_header=test_file_with_header,
            predict_file_with_header=predict_file_with_header)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_records = self.convert_examples_to_records(
            self.train_examples)
        self.dev_records = self.convert_examples_to_records(self.dev_examples)
        self.test_records = self.convert_examples_to_records(self.test_examples)
        self.predict_records = self.convert_examples_to_records(
            self.predict_examples)

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        has_warned = False
        with io.open(input_file, "r", encoding="UTF-8") as file:
            reader = csv.reader(file, delimiter="\t", quotechar=None)
            examples = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    ncol = len(line)
                    if self.if_file_with_header[phase]:
                        continue
                if phase != "predict":
                    if ncol == 1:
                        raise Exception(
                            "the %s file: %s only has one column but it is not a predict file"
                            % (phase, input_file))
                    elif ncol == 2:
                        example = InputExample(
                            guid=i, text_a=line[0], label=line[1])
                    elif ncol == 3:
                        example = InputExample(
                            guid=i,
                            text_a=line[0],
                            text_b=line[1],
                            label=line[2])
                    else:
                        raise Exception(
                            "the %s file: %s has too many columns (should <=3)"
                            % (phase, input_file))
                else:
                    if ncol == 1:
                        example = InputExample(guid=i, text_a=line[0])
                    elif ncol == 2:
                        if not has_warned:
                            logger.warning(
                                "the predict file: %s has 2 columns, as it is a predict file, the second one will be regarded as text_b"
                                % (input_file))
                            has_warned = True
                        example = InputExample(
                            guid=i, text_a=line[0], text_b=line[1])
                    else:
                        raise Exception(
                            "the predict file: %s has too many columns (should <=2)"
                            % (input_file))
                examples.append(example)
            return examples

    def convert_examples_to_records(self, examples):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.

        Returns:
            a list with all the examples record.
        """
        if not self.tokenizer:
            return []

        records = []
        for example in examples:
            record = self.tokenizer.encode(
                text=example.text_a,
                text_pair=example.text_b,
                max_seq_len=self.max_seq_len)
            if example.label:
                record["label"] = self.label_list.index(example.label)
            records.append(record)
        return records

    def get_train_records(self, shuffle=False):
        records = self.train_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_dev_records(self, shuffle=False):
        records = self.dev_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_test_records(self, shuffle=False):
        records = self.test_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_val_records(self, shuffle=False):
        records = self.get_dev_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_predict_records(self, shuffle=False):
        records = self.predict_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_phase_records(self, phase, shuffle=False):
        if phase == "train":
            return self.get_train_records(shuffle)
        elif phase == "dev":
            return self.get_dev_records(shuffle)
        elif phase == "test":
            return self.get_test_records(shuffle)
        elif phase == "val":
            return self.get_val_records(shuffle)
        elif phase == "predict":
            return self.get_predict_records(shuffle)
        else:
            raise ValueError("Invalid phase: %s" % phase)

    def get_phase_feed_list(self, phase):
        records = self.get_phase_records(phase)
        if records:
            feed_list = list(records[0].keys())
        else:
            if phase == "predict":
                feed_list = [
                    feed_name for feed_name in self.get_phase_feed_list("train")
                    if feed_name != "label"
                ]
            else:
                feed_list = [
                    feed_name for feed_name in self.get_phase_feed_list("train")
                ]
        return feed_list


class BaseSequenceLabelDataset(BaseNLPDataset):
    def convert_examples_to_records(self, examples):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.

        Returns:
            a list with all the examples record.
        """
        if not self.tokenizer:
            return []

        records = []
        for example in examples:
            tokens, labels = self._reseg_token_label(
                tokens=example.text_a.split("\002"),
                labels=example.label.split("\002"))
            record = self.tokenizer.encode(
                text=tokens, max_seq_len=self.max_seq_len)
            if labels:
                # Truncating
                if len(labels) > self.max_seq_len - 2:
                    labels = labels[0:(self.max_seq_len - 2)]
                # Filling
                no_entity_id = len(self.label_list) - 1
                label_ids = [no_entity_id] + [
                    self.label_list.index(label) for label in labels
                ] + [no_entity_id]
                # Padding
                label_ids = label_ids + [no_entity_id
                                         ] * (self.max_seq_len - len(label_ids))
                record["label"] = label_ids
            records.append(record)
        return records

    def _reseg_token_label(self, tokens, labels=None):
        if labels:
            if len(tokens) != len(labels):
                raise ValueError(
                    "The length of tokens must be same with labels")
            ret_tokens = []
            ret_labels = []
            for token, label in zip(tokens, labels):
                sub_token = self.tokenizer.tokenize(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                ret_labels.append(label)
                if len(sub_token) < 2:
                    continue
                sub_label = label
                if label.startswith("B-"):
                    sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token) - 1))

            if len(ret_tokens) != len(ret_labels):
                raise ValueError(
                    "The length of ret_tokens can't match with labels")
            return ret_tokens, ret_labels
        else:
            ret_tokens = []
            for token in tokens:
                sub_token = self.tokenizer.tokenize(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                if len(sub_token) < 2:
                    continue

            return ret_tokens, None
