#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import csv
import json
import numpy as np
import platform
import six
import sys
from collections import namedtuple

import paddle

from paddlehub.reader import tokenization
from paddlehub.common.logger import logger
from paddlehub.common.utils import sys_stdout_encoding
from paddlehub.dataset.dataset import InputExample
from .batching import pad_batch_data
import paddlehub as hub


class BaseReader(object):
    def __init__(self,
                 dataset,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 random_seed=None,
                 use_task_id=False):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.dataset = dataset
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = False
        self.use_task_id = use_task_id

        if self.use_task_id:
            self.task_id = 0

        np.random.seed(random_seed)

        # generate label map
        self.label_map = {}
        for index, label in enumerate(self.dataset.get_labels()):
            self.label_map[label] = index
        logger.info("Dataset label map = {}".format(self.label_map))

        self.current_example = 0
        self.current_epoch = 0

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.dataset.get_dev_examples()

    def get_val_examples(self):
        """Gets a collection of `InputExample`s for the val set."""
        return self.dataset.get_val_examples()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        return self.dataset.get_test_examples()

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self,
                                   example,
                                   max_seq_length,
                                   tokenizer,
                                   phase=None):
        """Converts a single `Example` into a single `Record`."""

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if example.text_b is not None:
            #if "text_b" in example._fields:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))

        if self.label_map:
            if example.label not in self.label_map:
                raise KeyError(
                    "example.label = {%s} not in label" % example.label)
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_id'])

        if phase != "predict":
            Record = namedtuple(
                'Record',
                ['token_ids', 'text_type_ids', 'position_ids', 'label_id'])

            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                label_id=label_id)
        else:
            Record = namedtuple('Record',
                                ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids)

        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer, phase)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records, phase)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records, phase)

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'val', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'val'/'dev', 'test']."
            )
        return self.num_examples[phase]

    def data_generator(self,
                       batch_size=1,
                       phase='train',
                       shuffle=True,
                       data=None):
        if phase == 'train':
            shuffle = True
            examples = self.get_train_examples()
            self.num_examples['train'] = len(examples)
        elif phase == 'val' or phase == 'dev':
            shuffle = False
            examples = self.get_dev_examples()
            self.num_examples['dev'] = len(examples)
        elif phase == 'test':
            shuffle = False
            examples = self.get_test_examples()
            self.num_examples['test'] = len(examples)
        elif phase == 'predict':
            shuffle = False
            examples = []
            seq_id = 0

            for item in data:
                # set label in order to run the program
                label = list(self.label_map.keys())[0]
                if len(item) == 1:
                    item_i = InputExample(
                        guid=seq_id, text_a=item[0], label=label)
                elif len(item) == 2:
                    item_i = InputExample(
                        guid=seq_id,
                        text_a=item[0],
                        text_b=item[1],
                        label=label)
                else:
                    raise ValueError(
                        "The length of input_text is out of handling, which must be 1 or 2!"
                    )
                examples.append(item_i)
                seq_id += 1
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test', 'predict']."
            )

        def wrapper():
            if shuffle:
                np.random.shuffle(examples)

            for batch_data in self._prepare_batch_data(
                    examples, batch_size, phase=phase):
                yield [batch_data]

        return wrapper


class ClassifyReader(BaseReader):
    def _pad_batch_records(self, batch_records, phase=None):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id,
            return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        if phase != "predict":
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])

            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_labels
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_labels
                ]
        else:
            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids
                ]
        return return_list


class SequenceLabelReader(BaseReader):
    def _pad_batch_records(self, batch_records, phase=None):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        if phase != "predict":
            batch_label_ids = [record.label_ids for record in batch_records]
            padded_label_ids = pad_batch_data(
                batch_label_ids,
                max_seq_len=self.max_seq_len,
                pad_idx=len(self.label_map) - 1)

            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, padded_label_ids, batch_seq_lens
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, padded_label_ids,
                    batch_seq_lens
                ]

        else:
            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_seq_lens
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_seq_lens
                ]

        return return_list

    def _reseg_token_label(self, tokens, tokenizer, phase, labels=None):
        if phase != "predict":
            if len(tokens) != len(labels):
                raise ValueError(
                    "The length of tokens must be same with labels")
            ret_tokens = []
            ret_labels = []
            for token, label in zip(tokens, labels):
                sub_token = tokenizer.tokenize(token)
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
                sub_token = tokenizer.tokenize(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                if len(sub_token) < 2:
                    continue

            return ret_tokens

    def _convert_example_to_record(self,
                                   example,
                                   max_seq_length,
                                   tokenizer,
                                   phase=None):

        tokens = tokenization.convert_to_unicode(example.text_a).split(u"")

        if phase != "predict":
            labels = tokenization.convert_to_unicode(example.label).split(u"")
            tokens, labels = self._reseg_token_label(
                tokens=tokens, labels=labels, tokenizer=tokenizer, phase=phase)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            position_ids = list(range(len(token_ids)))
            text_type_ids = [0] * len(token_ids)
            no_entity_id = len(self.label_map) - 1
            label_ids = [no_entity_id
                         ] + [self.label_map[label]
                              for label in labels] + [no_entity_id]

            Record = namedtuple(
                'Record',
                ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                label_ids=label_ids)
        else:
            tokens = self._reseg_token_label(
                tokens=tokens, tokenizer=tokenizer, phase=phase)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            position_ids = list(range(len(token_ids)))
            text_type_ids = [0] * len(token_ids)

            Record = namedtuple('Record',
                                ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
            )

        return record


class LACClassifyReader(object):
    def __init__(self, dataset, vocab_path):
        self.dataset = dataset
        self.lac = hub.Module(name="lac")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=False)
        self.vocab = self.tokenizer.vocab
        self.feed_key = list(
            self.lac.processor.data_format(
                sign_name="lexical_analysis").keys())[0]

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'val', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'val'/'dev', 'test']."
            )
        return self.num_examples[phase]

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.dataset.get_dev_examples()

    def get_val_examples(self):
        """Gets a collection of `InputExample`s for the val set."""
        return self.dataset.get_val_examples()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        return self.dataset.get_test_examples()

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def data_generator(self,
                       batch_size=1,
                       phase="train",
                       shuffle=False,
                       data=None):
        if phase == "train":
            shuffle = True
            data = self.dataset.get_train_examples()
            self.num_examples['train'] = len(data)
        elif phase == "test":
            shuffle = False
            data = self.dataset.get_test_examples()
            self.num_examples['test'] = len(data)
        elif phase == "val" or phase == "dev":
            shuffle = False
            data = self.dataset.get_dev_examples()
            self.num_examples['dev'] = len(data)
        elif phase == "predict":
            data = data
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")

        def preprocess(text):
            data_dict = {self.feed_key: [text]}
            processed = self.lac.lexical_analysis(data=data_dict)
            processed = [
                self.vocab[word] for word in processed[0]['word']
                if word in self.vocab
            ]
            if len(processed) == 0:
                if six.PY2:
                    text = text.encode(sys_stdout_encoding())
                logger.warning(
                    "The words in text %s can't be found in the vocabulary." %
                    (text))
            return processed

        def _data_reader():
            if shuffle:
                np.random.shuffle(data)

            if phase == "predict":
                for text in data:
                    text = preprocess(text)
                    if not text:
                        continue
                    yield (text, )
            else:
                for item in data:
                    text = preprocess(item.text_a)
                    if not text:
                        continue
                    yield (text, item.label)

        return paddle.batch(_data_reader, batch_size=batch_size)


class MultiLabelClassifyReader(BaseReader):
    def _pad_batch_records(self, batch_records, phase=None):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len,
            return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        if phase != "predict":
            batch_labels_ids = [record.label_ids for record in batch_records]
            num_label = len(self.dataset.get_labels())
            batch_labels = np.array(batch_labels_ids).astype("int64").reshape(
                [-1, num_label])

            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_labels
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_labels
                ]
        else:
            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids
                ]
        return return_list

    def _convert_example_to_record(self,
                                   example,
                                   max_seq_length,
                                   tokenizer,
                                   phase=None):
        """Converts a single `Example` into a single `Record`."""

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if example.text_b is not None:
            #if "text_b" in example._fields:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))

        label_ids = []
        if phase == "predict":
            label_ids = [0, 0, 0, 0, 0, 0]
        else:
            for label in example.label:
                label_ids.append(int(label))

        if phase != "predict":
            Record = namedtuple(
                'Record',
                ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])

            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                label_ids=label_ids)
        else:
            Record = namedtuple('Record',
                                ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids)

        return record


if __name__ == '__main__':
    pass
