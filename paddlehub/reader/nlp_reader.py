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

import collections
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
from .batching import pad_batch_data, prepare_batch_data
import paddlehub as hub


class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 dataset=None,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 random_seed=None,
                 use_task_id=False,
                 sp_model_path=None,
                 word_dict_path=None,
                 in_tokens=False):
        self.max_seq_len = max_seq_len
        if sp_model_path and word_dict_path:
            self.tokenizer = tokenization.WSSPTokenizer(
                vocab_path, sp_model_path, word_dict_path, ws=True, lower=True)
        else:
            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.dataset = dataset
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.use_task_id = use_task_id

        if self.use_task_id:
            self.task_id = 0

        np.random.seed(random_seed)

        # generate label map
        self.label_map = {}
        if self.dataset:
            for index, label in enumerate(self.dataset.get_labels()):
                self.label_map[label] = index
            logger.info("Dataset label map = {}".format(self.label_map))
        else:
            logger.info("Dataset is None! label map = {}".format(
                self.label_map))

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

    def _pad_batch_records(self, batch_records, phase):
        raise NotImplementedError

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
        if phase != 'predict' and not self.dataset:
            raise ValueError("The dataset is None ! It isn't allowed.")
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
                if self.dataset:
                    label = list(self.label_map.keys())[0]
                else:
                    label = 0
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
    def __init__(self,
                 vocab_path,
                 dataset=None,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 random_seed=None,
                 use_task_id=False,
                 sp_model_path=None,
                 word_dict_path=None,
                 in_tokens=False):
        super(SequenceLabelReader, self).__init__(
            vocab_path=vocab_path,
            dataset=dataset,
            label_map_config=label_map_config,
            max_seq_len=max_seq_len,
            do_lower_case=do_lower_case,
            random_seed=random_seed,
            use_task_id=use_task_id,
            sp_model_path=sp_model_path,
            word_dict_path=word_dict_path,
            in_tokens=in_tokens)
        if sp_model_path and word_dict_path:
            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_path,
                do_lower_case=do_lower_case,
                use_sentence_piece_vocab=True)

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
    def __init__(self, vocab_path, dataset=None, in_tokens=False):
        self.dataset = dataset
        self.lac = hub.Module(name="lac")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=False)
        self.vocab = self.tokenizer.vocab
        self.feed_key = list(
            self.lac.processor.data_format(
                sign_name="lexical_analysis").keys())[0]

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        self.in_tokens = in_tokens

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
        if phase != "predict" and not self.dataset:
            raise ValueError("The dataset is None and it isn't allowed.")
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


class RegressionReader(BaseReader):
    def __init__(self,
                 vocab_path,
                 dataset=None,
                 label_map_config=None,
                 max_seq_len=128,
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
        self.label_map = {}  # Unlike BaseReader, it's not filled

        self.current_example = 0
        self.current_epoch = 0

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

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
            # the only diff with ClassifyReader: astype("float32")
            batch_labels = np.array(batch_labels).astype("float32").reshape(
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

    def data_generator(self,
                       batch_size=1,
                       phase='train',
                       shuffle=True,
                       data=None):
        if phase != 'predict' and not self.dataset:
            raise ValueError("The dataset is none and it's not allowed.")
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
                label = -1  # different from BaseReader
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


class Features(object):
    """A single set of features of squad_data."""

    def __init__(
            self,
            unique_id,
            example_index,
            doc_span_index,
            tokens,
            token_to_orig_map,
            token_is_max_context,
            token_ids,
            position_ids,
            text_type_ids,
            start_position=None,
            end_position=None,
            is_impossible=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.text_type_ids = text_type_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __repr__(self):
        s = ""
        s += "unique_id: %s " % self.unique_id
        s += "example_index: %s " % self.example_index
        s += "start_position: %s " % self.start_position
        s += "end_position: %s " % self.end_position
        s += "is_impossible: %s " % self.is_impossible
        # s += "tokens: %s" % self.tokens
        # s += "token_to_orig_map %s" % self.token_to_orig_map
        return s


class ReadingComprehensionReader(BaseReader):
    def __init__(self,
                 dataset,
                 vocab_path,
                 do_lower_case=True,
                 max_seq_len=512,
                 doc_stride=128,
                 max_query_length=64,
                 random_seed=None,
                 use_task_id=False):
        self.dataset = dataset
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.use_task_id = use_task_id
        self.in_tokens = False
        # self.all_examples[phase] and self.all_features[phase] will be used
        # in write_prediction in reading_comprehension_task
        self.all_features = {"train": [], "dev": [], "test": [], "predict": []}
        self.all_examples = {"train": [], "dev": [], "test": [], "predict": []}

        np.random.seed(random_seed)

        self.vocab = self.tokenizer.vocab
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]

        self.current_train_example = 0

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def _pad_batch_records(self, batch_records, phase):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_unique_ids = [record.unique_id for record in batch_records]
        batch_unique_ids = np.array(batch_unique_ids).astype("int64").reshape(
            [-1, 1])

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            max_seq_len=self.max_seq_len)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len)
        padded_position_ids = pad_batch_data(
            batch_position_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len)

        if phase != "predict":
            batch_start_position = [
                record.start_position for record in batch_records
            ]
            batch_end_position = [
                record.end_position for record in batch_records
            ]
            batch_start_position = np.array(batch_start_position).astype(
                "int64").reshape([-1, 1])
            batch_end_position = np.array(batch_end_position).astype(
                "int64").reshape([-1, 1])

            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_unique_ids, batch_start_position,
                batch_end_position
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_unique_ids,
                    batch_start_position, batch_end_position
                ]

        else:
            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_unique_ids
            ]
            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_unique_ids
                ]
        return return_list

    def _prepare_batch_data(self, records, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, record in enumerate(records):
            if phase == "train":
                self.current_example = index
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

    def data_generator(self,
                       batch_size=1,
                       phase='train',
                       shuffle=False,
                       data=None):
        # we need all_examples and  all_features in write_prediction in reading_comprehension_task
        # we can also use all_examples and all_features to avoid duplicate long-time preprocessing
        examples = None
        if self.all_examples[phase]:
            examples = self.all_examples[phase]
        else:
            if phase == 'train':
                examples = self.get_train_examples()
            elif phase == 'dev':
                examples = self.get_dev_examples()
            elif phase == 'test':
                examples = self.get_test_examples()
            elif phase == 'predict':
                examples = data
            else:
                raise ValueError(
                    "Unknown phase, which should be in ['train', 'dev', 'test', 'predict']."
                )
            self.all_examples[phase] = examples
        shuffle = True if phase == 'train' else False

        # As reading comprehension task will divide a long context into several doc_spans and then get multiple features
        # To get the real total steps, we need to know the features' length
        # So we use _convert_examples_to_records rather than _convert_example_to_record in this task
        if self.all_features[phase]:
            features = self.all_features[phase]
        else:
            features = self._convert_examples_to_records(
                examples, self.max_seq_len, self.tokenizer, phase)
            self.all_features[phase] = features

        # self.num_examples["train"] use in strategy.py to show the total steps,
        # we need to cover it with correct len(features)
        self.num_examples[phase] = len(features)

        def wrapper():
            if shuffle:
                np.random.shuffle(features)

            for batch_data in self._prepare_batch_data(
                    features, batch_size, phase=phase):
                yield [batch_data]

        return wrapper

    def _convert_examples_to_records(self,
                                     examples,
                                     max_seq_length,
                                     tokenizer,
                                     phase=None):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        unique_id = 1000000000

        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)
            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            is_impossible = example.is_impossible if hasattr(
                example, "is_impossible") else False

            if phase != "predict" and is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if phase != "predict" and not is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position +
                                                         1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position,
                 tok_end_position) = self.improve_answer_span(
                     all_doc_tokens, tok_start_position, tok_end_position,
                     tokenizer, example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                text_type_ids = []
                tokens.append("[CLS]")
                text_type_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    text_type_ids.append(0)
                tokens.append("[SEP]")
                text_type_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(
                        tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self.check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    text_type_ids.append(1)
                tokens.append("[SEP]")
                text_type_ids.append(1)

                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                position_ids = list(range(len(token_ids)))
                start_position = None
                end_position = None
                if phase != "predict" and not is_impossible:
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
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if phase != "predict" and is_impossible:
                    start_position = 0
                    end_position = 0

                feature = Features(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    token_ids=token_ids,
                    position_ids=position_ids,
                    text_type_ids=text_type_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                features.append(feature)

                unique_id += 1

        return features

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


if __name__ == '__main__':
    pass
