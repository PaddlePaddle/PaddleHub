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
import collections

from tqdm import tqdm
from paddlehub.dataset import InputExample, BaseDataset
from paddlehub.common.logger import logger
from paddlehub.tokenizer import CustomTokenizer, BertTokenizer
import numpy as np


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
                 max_seq_len=128):
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
        self._train_records = None
        self._dev_records = None
        self._test_records = None
        self._predict_records = None

    @property
    def train_records(self):
        if not self._train_records:
            examples = self.train_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the train set...")
            self._train_records = self._convert_examples_to_records(
                examples, phase="train")
        return self._train_records

    @property
    def dev_records(self):
        if not self._dev_records:
            examples = self.dev_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the dev set...")
            self._dev_records = self._convert_examples_to_records(
                examples, phase="dev")
        return self._dev_records

    @property
    def test_records(self):
        if not self._test_records:
            examples = self.test_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the test set...")
            self._test_records = self._convert_examples_to_records(
                examples, phase="test")
        return self._test_records

    @property
    def predict_records(self):
        if not self._predict_records:
            examples = self.predict_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the predict set...")
            self._predict_records = self._convert_examples_to_records(
                examples, phase="predict")
        return self._predict_records

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

    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data example, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".


        Returns:
            a list with all the examples record.
        """

        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a,
                    text_pair=example.text_b,
                    max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue
                if example.label:
                    record["label"] = self.label_list.index(
                        example.label) if self.label_list else float(
                            example.label)
                records.append(record)
                process_bar.update(1)
        return records

    def get_train_records(self, shuffle=False):
        return self.get_records("train", shuffle=shuffle)

    def get_dev_records(self, shuffle=False):
        return self.get_records("dev", shuffle=shuffle)

    def get_test_records(self, shuffle=False):
        return self.get_records("test", shuffle=shuffle)

    def get_val_records(self, shuffle=False):
        return self.get_records("val", shuffle=shuffle)

    def get_predict_records(self, shuffle=False):
        return self.get_records("predict", shuffle=shuffle)

    def get_records(self, phase, shuffle=False):
        if phase == "train":
            records = self.train_records
        elif phase == "dev":
            records = self.dev_records
        elif phase == "test":
            records = self.test_records
        elif phase == "val":
            records = self.dev_records
        elif phase == "predict":
            records = self.predict_records
        else:
            raise ValueError("Invalid phase: %s" % phase)

        if shuffle:
            np.random.shuffle(records)
        return records

    def get_feed_list(self, phase):
        records = self.get_records(phase)
        if records:
            feed_list = list(records[0].keys())
        else:
            feed_list = []
        return feed_list

    def batch_records_generator(self,
                                phase,
                                batch_size,
                                shuffle=True,
                                pad_to_batch_max_seq_len=False):
        """ generate a batch of records, usually used in dynamic graph mode.

        Args:
            phase (str): the dataset phase, can be "train", "dev", "val", "test" or "predict".
            batch_size (int): the data batch size
            shuffle (bool): if set to True, will shuffle the dataset.
            pad_to_batch_max_seq_len (bool): if set to True, will dynamically pad to the max sequence length of the batch data.
                                             Only recommended to set to True when the model has used RNN.
        """
        records = self.get_records(phase, shuffle=shuffle)

        batch_records = []
        batch_lens = []
        for record in records:
            batch_records.append(record)
            if pad_to_batch_max_seq_len:
                # This may reduce the processing speed
                tokens_wo_pad = [
                    token for token in self.tokenizer.decode(
                        record, only_convert_to_tokens=True)
                    if token != self.tokenizer.pad_token
                ]
                batch_lens.append(len(tokens_wo_pad))
            if len(batch_records) == batch_size:
                if pad_to_batch_max_seq_len:
                    # This may reduce the processing speed.
                    batch_max_seq_len = max(batch_lens)
                    for record in batch_records:
                        for key, value in record.items():
                            if isinstance(value, list):
                                # This may not be universal
                                record[key] = value[:batch_max_seq_len]
                rev_batch_records = {
                    key: [record[key] for record in batch_records]
                    for key in batch_records[0]
                }
                yield rev_batch_records
                batch_records = []
                batch_lens = []

        if batch_records:
            if pad_to_batch_max_seq_len:
                # This may reduce the processing speed.
                batch_max_seq_len = max(batch_lens)
                for record in batch_records:
                    for key in record.keys():
                        if isinstance(record[key], list):
                            record[key] = record[key][:batch_max_seq_len]
            rev_batch_records = {
                key: [record[key] for record in batch_records]
                for key in batch_records[0]
            }
            yield rev_batch_records


class TextClassificationDataset(BaseNLPDataset):
    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data example, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".


        Returns:
            a list with all the examples record.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a,
                    text_pair=example.text_b,
                    max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue
                if example.label:
                    record["label"] = self.label_list.index(example.label)
                records.append(record)
                process_bar.update(1)
        return records


class RegressionDataset(BaseNLPDataset):
    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data example, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".

        Returns:
            a list with all the examples record.
        """

        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a,
                    text_pair=example.text_b,
                    max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue
                if example.label:
                    record["label"] = float(example.label)
                records.append(record)
                process_bar.update(1)
        return records


class GenerationDataset(BaseNLPDataset):
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
                 max_seq_len=128,
                 split_char="\002",
                 start_token="<s>",
                 end_token="</s>",
                 unk_token="<unk>"):
        self.split_char = split_char
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        super(GenerationDataset, self).__init__(
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
            predict_file_with_header=predict_file_with_header,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)

    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data example, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".

        Returns:
            a list with all the examples record.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a.split(self.split_char),
                    text_pair=example.text_b.split(self.split_char)
                    if example.text_b else None,
                    max_seq_len=self.max_seq_len)
                if example.label:
                    expand_label = [self.start_token] + example.label.split(
                        self.split_char)[:self.max_seq_len - 2] + [
                            self.end_token
                        ]
                    expand_label_id = [
                        self.label_index.get(label,
                                             self.label_index[self.unk_token])
                        for label in expand_label
                    ]
                    record["label"] = expand_label_id[1:] + [
                        self.label_index[self.end_token]
                    ] * (self.max_seq_len - len(expand_label) + 1)
                    record["dec_input"] = expand_label_id[:-1] + [
                        self.label_index[self.end_token]
                    ] * (self.max_seq_len - len(expand_label) + 1)
                records.append(record)
                process_bar.update(1)
        return records


class SeqLabelingDataset(BaseNLPDataset):
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
                 max_seq_len=128,
                 split_char="\002",
                 no_entity_label="O"):
        self.no_entity_label = no_entity_label
        self.split_char = split_char

        super(SeqLabelingDataset, self).__init__(
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
            predict_file_with_header=predict_file_with_header,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)

    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".

        Returns:
            a list with all the examples record.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                tokens, labels = self._reseg_token_label(
                    tokens=example.text_a.split(self.split_char),
                    labels=example.label.split(self.split_char))
                record = self.tokenizer.encode(
                    text=tokens, max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue
                if labels:
                    record["label"] = []
                    tokens_with_specical_token = self.tokenizer.decode(
                        record, only_convert_to_tokens=True)
                    tokens_index = 0
                    for token in tokens_with_specical_token:
                        if tokens_index < len(
                                tokens) and token == tokens[tokens_index]:
                            record["label"].append(
                                self.label_list.index(labels[tokens_index]))
                            tokens_index += 1
                        else:
                            record["label"].append(
                                self.label_list.index(self.no_entity_label))
                records.append(record)
                process_bar.update(1)
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


class MultiLabelDataset(BaseNLPDataset):
    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".

        Returns:
            a list with all the examples record.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a,
                    text_pair=example.text_b,
                    max_seq_len=self.max_seq_len)

                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue

                if example.label:
                    record["label"] = [int(label) for label in example.label]
                records.append(record)
                process_bar.update(1)
        return records


class MRCDataset(BaseNLPDataset):
    def __init__(
            self,
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
            max_seq_len=128,
            max_query_len=64,
            doc_stride=128,
    ):

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
            predict_file_with_header=predict_file_with_header,
        )

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_query_len = max_query_len
        self._DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        self.doc_stride = doc_stride
        self._Feature = collections.namedtuple("Feature", [
            "unique_id",
            "example_index",
            "doc_span_index",
            "tokens",
            "token_to_orig_map",
            "token_is_max_context",
        ])
        self.special_tokens_num, self.special_tokens_num_before_doc = self._get_special_tokens_num(
        )

        self._train_records = None
        self._dev_records = None
        self._test_records = None
        self._predict_records = None
        self._train_features = None
        self._dev_features = None
        self._test_features = None
        self._predict_features = None

    @property
    def train_records(self):
        if not self._train_records:
            examples = self.train_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the train set...")
            self._train_records, self._train_features = self._convert_examples_to_records_and_features(
                examples, "train")
        return self._train_records

    @property
    def dev_records(self):
        if not self._dev_records:
            examples = self.dev_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the dev set...")
            self._dev_records, self._dev_features = self._convert_examples_to_records_and_features(
                examples, "dev")
        return self._dev_records

    @property
    def test_records(self):
        if not self._test_records:
            examples = self.test_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the test set...")
            self._test_records, self._test_features = self._convert_examples_to_records_and_features(
                examples, "test")
        return self._test_records

    @property
    def predict_records(self):
        if not self._predict_records:
            examples = self.predict_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the predict set...")
            self._predict_records, self._predict_features = self._convert_examples_to_records_and_features(
                examples, "predict")
        return self._predict_records

    @property
    def train_features(self):
        if not self._train_features:
            examples = self.train_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the train set...")
            self._train_records, self._train_features = self._convert_examples_to_records_and_features(
                examples, "train")
        return self._train_features

    @property
    def dev_features(self):
        if not self._dev_features:
            examples = self.dev_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the dev set...")
            self._dev_records, self._dev_features = self._convert_examples_to_records_and_features(
                examples, "dev")
        return self._dev_features

    @property
    def test_features(self):
        if not self._test_features:
            examples = self.test_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the test set...")
            self._test_records, self._test_features = self._convert_examples_to_records_and_features(
                examples, "test")
        return self._test_features

    @property
    def predict_features(self):
        if not self._predict_features:
            examples = self.predict_examples
            if not self.tokenizer or not examples:
                return []
            logger.info("Processing the predict set...")
            self._predict_records, self._predict_features = self._convert_examples_to_records_and_features(
                examples, "predict")
        return self._predict_features

    def _get_special_tokens_num(self):
        if not self.tokenizer:
            return None, None
        # We must have a pad token, so we can use it to make fake text.
        fake_question = [self.tokenizer.pad_token]
        fake_answer = [self.tokenizer.pad_token]
        special_tokens_num = 0
        special_tokens_num_before_doc = 0
        seen_pad_num = 0
        fake_record = self.tokenizer.encode(fake_question, fake_answer)
        fake_tokens_with_special_tokens = self.tokenizer.decode(
            fake_record, only_convert_to_tokens=True)
        for token in fake_tokens_with_special_tokens:
            if token == self.tokenizer.pad_token:
                seen_pad_num += 1
                if seen_pad_num > 2:
                    # The third pad_token is added by padding
                    break
            else:
                special_tokens_num += 1
                if seen_pad_num < 2:
                    # The second pad_token is the fake_answer
                    special_tokens_num_before_doc += 1
        return special_tokens_num, special_tokens_num_before_doc

    def _convert_examples_to_records_and_features(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".

        Returns:
            a list with all the examples record.
        """
        features = []
        records = []
        unique_id = 1000000000

        with tqdm(total=len(examples)) as process_bar:
            for (example_index, example) in enumerate(examples):
                # Tokenize question_text
                query_tokens = self.tokenizer.tokenize(example.question_text)
                if len(query_tokens) > self.max_query_len:
                    query_tokens = query_tokens[0:self.max_query_len]

                # Tokenize doc_tokens and get token-sub_token position map
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(example.doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    sub_tokens = self.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                # Update the answer position to the new sub_token position
                tok_start_position = None
                tok_end_position = None
                is_impossible = example.is_impossible if hasattr(
                    example, "is_impossible") else False

                if phase != "predict" and is_impossible:
                    tok_start_position = -1
                    tok_end_position = -1
                if phase != "predict" and not is_impossible:
                    tok_start_position = orig_to_tok_index[
                        example.start_position]
                    if example.end_position < len(example.doc_tokens) - 1:
                        tok_end_position = orig_to_tok_index[
                            example.end_position + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position,
                     tok_end_position) = self.improve_answer_span(
                         all_doc_tokens, tok_start_position, tok_end_position,
                         self.tokenizer, example.orig_answer_text)

                # We can have documents that are longer than the maximum sequence length.
                # To deal with this we do a sliding window approach, where we take chunks
                # of the up to our max length with a stride of `doc_stride`.
                # if hasattr(self.tokenizer, "num_special_tokens_to_add"):
                max_tokens_for_doc = self.max_seq_len - len(
                    query_tokens) - self.special_tokens_num

                doc_spans = []
                start_offset = 0
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_tokens_for_doc:
                        length = max_tokens_for_doc
                    doc_spans.append(
                        self._DocSpan(start=start_offset, length=length))
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, self.doc_stride)

                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    # Update the start_position and end_position to doc_span
                    start_position = None
                    end_position = None
                    if phase != "predict":
                        if is_impossible:
                            start_position = 0
                            end_position = 0
                        else:
                            # For training, if our document chunk does not contain an annotation
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span.start
                            doc_end = doc_span.start + doc_span.length - 1
                            out_of_span = False
                            if not (tok_start_position >= doc_start
                                    and tok_end_position <= doc_end):
                                out_of_span = True
                            if out_of_span:
                                start_position = 0
                                end_position = 0
                            else:
                                doc_offset = len(
                                    query_tokens
                                ) + self.special_tokens_num_before_doc
                                start_position = tok_start_position - doc_start + doc_offset
                                end_position = tok_end_position - doc_start + doc_offset

                    record = self.tokenizer.encode(
                        text=query_tokens,
                        text_pair=all_doc_tokens[doc_span.start:doc_span.start +
                                                 doc_span.length],
                        max_seq_len=self.max_seq_len)
                    record["start_position"] = start_position
                    record["end_position"] = end_position
                    record["unique_id"] = unique_id
                    records.append(record)

                    # The other information is saved in feature, which is helpful in postprocessing.
                    # The bridge with record and feature is unique_id.
                    tokens = self.tokenizer.decode(
                        record, only_convert_to_tokens=True)
                    token_to_orig_map = {}
                    token_is_max_context = {}
                    doc_token_start = len(
                        query_tokens) + self.special_tokens_num_before_doc
                    for i in range(doc_span.length):
                        # split_token_index: the doc token position in doc after tokenize
                        # doc_token_index: the doc token position in record after encode
                        split_token_index = doc_span.start + i
                        doc_token_index = doc_token_start + i
                        token_to_orig_map[doc_token_index] = tok_to_orig_index[
                            split_token_index]
                        is_max_context = self.check_is_max_context(
                            doc_spans, doc_span_index, split_token_index)
                        token_is_max_context[doc_token_index] = is_max_context

                    feature = self._Feature(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                    )
                    features.append(feature)

                    unique_id += 1
                process_bar.update(1)

        return records, features

    def improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                            orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context,
                        num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def get_features(self, phase):
        if phase == "train":
            return self.train_features
        elif phase == "dev":
            return self.dev_features
        elif phase == "test":
            return self.test_features
        elif phase == "val":
            return self.dev_features
        elif phase == "predict":
            return self.predict_features
        else:
            raise ValueError("Invalid phase: %s" % phase)


class TextMatchingDataset(BaseNLPDataset):
    """
    Text Matching DataSet base class, including point_wise and pair_wise mode.
    """

    def __init__(self,
                 base_path,
                 is_pair_wise=False,
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
                 max_seq_len=128):
        """
        Args:
            base_path(str): The directory to dataset, which includes train, dev, test or prediction data.
            is_pair_wise(bool): The dataset is pair wise or not. Default as False.
            train_file(str): The train data file name of the dataset.
            dev_file(str): The development data file name of the dataset. It is optional.
            test_file(str): The test data file name of the dataset. It is optional.
            predict_file(str): The prediction data file name of the dataset. It is optional.
            label_file(str): It is a file name, which contains labels of the dataset. If label file not label_list.
            label_list(list): It is the labels of the dataset.
            train_file_with_header(bool): The train file is with introduction of the file in the first line or not. Default as False.
            dev_file_with_header(bool): The development file is with introduction of the file in the first line or not. Default as False.
            test_file_with_header(bool): The test file is with introduction of the file in the first line or not. Default as False.
            tokenizer(object): It should be hub.BertTokenizer or hub.CustomTokenizer, which tokenizes the text and encodes the data as model needed.
            max_seq_len(int): It will limit the total sequence returned so that it has a maximum length.
        """
        self.is_pair_wise = is_pair_wise
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
        self._train_records = None
        self._dev_records = None
        self._test_records = None
        self._predict_records = None

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        has_warned = False
        if self.is_pair_wise:
            InputExample = collections.namedtuple(
                'Example', ['guid', 'text_a', 'text_b', 'text_c', 'label'])
        else:
            InputExample = collections.namedtuple(
                'Example', ['guid', 'text_a', 'text_b', 'label'])
        with io.open(input_file, "r", encoding="UTF-8") as file:
            reader = csv.reader(file, delimiter="\t", quotechar=None)
            examples = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    ncol = len(line)
                    if self.if_file_with_header[phase]:
                        continue
                if phase != "predict":
                    if ncol <= 2:
                        raise Exception(
                            "the %s file (%s) is illegal. The %s file of a text matching task should have 3 or 4 columns. Instead, it has only %d columns. Please check the data."
                            % (phase, input_file, phase, ncol))
                    elif ncol == 3:
                        if self.is_pair_wise:
                            raise Exception(
                                "the %s file (%s) is illegal. The %s file of a pair-wise text matching task should have 4 columns. Instead, it has only %d columns. Please check the data."
                                % (phase, input_file, phase, ncol))
                        else:
                            example = InputExample(
                                guid=i,
                                text_a=line[0],
                                text_b=line[1],
                                label=line[2])
                    elif ncol == 4 and self.is_pair_wise:
                        example = InputExample(
                            guid=i,
                            text_a=line[0],
                            text_b=line[1],
                            text_c=line[2],
                            label=line[3])
                    else:
                        raise Exception(
                            "the %s file (%s) has too many columns (should <=4), is illegal."
                            % (phase, input_file))
                else:
                    if ncol == 1:
                        raise Exception(
                            "the %s file (%s) is illegal. The %s file of a text matching task should have 2 columns. Instead, it has only one column. Please check the data."
                            % (phase, input_file, phase))
                    elif ncol == 2:
                        if self.is_pair_wise:
                            raise Exception(
                                "the %s file (%s) is illegal. The %s file of a pair-wise text matching task should have 3 columns. Instead, it has only 2 columns. Please check the data."
                                % (phase, input_file, phase))
                            example = InputExample(
                                guid=i,
                                text_a=line[0],
                                text_b=line[1],
                                text_c=None,
                                label=None)
                        else:
                            example = InputExample(
                                guid=i,
                                text_a=line[0],
                                text_b=line[1],
                                label=None)
                    elif ncol == 3:
                        if self.is_pair_wise:
                            example = InputExample(
                                guid=i,
                                text_a=line[0],
                                text_b=line[1],
                                text_c=line[2],
                                label=None)
                        else:
                            raise Exception(
                                "the %s file (%s) is illegal. The %s file of a pair-wise text matching task should have 3 columns. Instead, it has only 2 columns. Please check the data."
                                % (phase, input_file, phase))
                    else:
                        raise Exception(
                            "the predict file: %s has too many columns." %
                            (input_file))
                examples.append(example)
            return examples

    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model needs.

        Args:
            examples (list): the data example, returned by _read_file.
            phase(str): train, dev, test or predict.

        Returns:
            a list with all the records, which will be feeded to the prpgram.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record_a = self.tokenizer.encode(
                    text=example.text_a, max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record_a:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue

                record_b = self.tokenizer.encode(
                    text=example.text_b, max_seq_len=self.max_seq_len)
                if not record_b:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_b)
                    continue

                record = {}

                if isinstance(self.tokenizer, CustomTokenizer):
                    record = {
                        'text': record_a['text'],
                        'text_2': record_b['text'],
                        'seq_len': record_a['seq_len'],
                        'seq_len_2': record_b['seq_len'],
                    }

                    if self.is_pair_wise and example.text_c:
                        record_c = self.tokenizer.encode(
                            text=example.text_c, max_seq_len=self.max_seq_len)
                        if not record_c:
                            logger.info(
                                "The text %s has been dropped as it has no words in the vocab after tokenization."
                                % example.text_c)
                            continue

                        record['text_3'] = record_c['text']
                        record['seq_len_3'] = record_c['seq_len']
                elif isinstance(self.tokenizer, BertTokenizer):
                    record = {
                        # text_1
                        'input_ids': record_a['input_ids'],
                        'segment_ids': record_a['segment_ids'],
                        'input_mask': record_a['input_mask'],
                        'position_ids': record_a['position_ids'],
                        'seq_len': record_a['seq_len'],
                        # text_2
                        'input_ids_2': record_b['input_ids'],
                        'segment_ids_2': record_b['segment_ids'],
                        'input_mask_2': record_b['input_mask'],
                        'position_ids_2': record_b['position_ids'],
                        'seq_len_2': record_b['seq_len'],
                    }
                    if self.is_pair_wise and example.text_c:
                        # text_3
                        record_c = self.tokenizer.encode(
                            text=example.text_c, max_seq_len=self.max_seq_len)
                        record['input_ids_3'] = record_c['input_ids']
                        record['segment_ids_3'] = record_c['segment_ids']
                        record['input_mask_3'] = record_c['input_mask']
                        record['position_ids_3'] = record_c['position_ids']
                        record['seq_len_3'] = record_c['seq_len']
                else:
                    raise Exception(
                        "Unknown Tokenizer %s! Please redefine the _convert_examples_to_records method of TextMatchingDataset."
                        % self.tokenizer.__name__)

                if example.label:
                    record['label'] = self.label_list.index(example.label)

                records.append(record)
                process_bar.update(1)

        return records
