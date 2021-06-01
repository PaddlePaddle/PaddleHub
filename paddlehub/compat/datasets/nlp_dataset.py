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

import io
import csv
import collections

import numpy as np
from tqdm import tqdm

from paddlehub.compat.datasets.base_dataset import InputExample, BaseDataset
from paddlehub.utils.log import logger
from paddlehub.text.tokenizer import CustomTokenizer
from paddlehub.text.bert_tokenizer import BertTokenizer


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
            logger.info('Processing the train set...')
            self._train_records = self._convert_examples_to_records(examples, phase='train')
        return self._train_records

    @property
    def dev_records(self):
        if not self._dev_records:
            examples = self.dev_examples
            if not self.tokenizer or not examples:
                return []
            logger.info('Processing the dev set...')
            self._dev_records = self._convert_examples_to_records(examples, phase='dev')
        return self._dev_records

    @property
    def test_records(self):
        if not self._test_records:
            examples = self.test_examples
            if not self.tokenizer or not examples:
                return []
            logger.info('Processing the test set...')
            self._test_records = self._convert_examples_to_records(examples, phase='test')
        return self._test_records

    @property
    def predict_records(self):
        if not self._predict_records:
            examples = self.predict_examples
            if not self.tokenizer or not examples:
                return []
            logger.info('Processing the predict set...')
            self._predict_records = self._convert_examples_to_records(examples, phase='predict')
        return self._predict_records

    def _read_file(self, input_file, phase=None):
        '''Reads a tab separated value file.'''
        has_warned = False
        with io.open(input_file, 'r', encoding='UTF-8') as file:
            reader = csv.reader(file, delimiter='\t', quotechar=None)
            examples = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    ncol = len(line)
                    if self.if_file_with_header[phase]:
                        continue
                if phase != 'predict':
                    if ncol == 1:
                        raise Exception(
                            'the %s file: %s only has one column but it is not a predict file' % (phase, input_file))
                    elif ncol == 2:
                        example = InputExample(guid=i, text_a=line[0], label=line[1])
                    elif ncol == 3:
                        example = InputExample(guid=i, text_a=line[0], text_b=line[1], label=line[2])
                    else:
                        raise Exception('the %s file: %s has too many columns (should <=3)' % (phase, input_file))
                else:
                    if ncol == 1:
                        example = InputExample(guid=i, text_a=line[0])
                    elif ncol == 2:
                        if not has_warned:
                            logger.warning(
                                'the predict file: %s has 2 columns, as it is a predict file, the second one will be regarded as text_b'
                                % (input_file))
                            has_warned = True
                        example = InputExample(guid=i, text_a=line[0], text_b=line[1])
                    else:
                        raise Exception('the predict file: %s has too many columns (should <=2)' % (input_file))
                examples.append(example)
            return examples

    def _convert_examples_to_records(self, examples, phase):
        '''
        Returns a list[dict] including all the input information what the model need.
        Args:
            examples (list): the data example, returned by _read_file.
            phase (str): the processing phase, can be 'train' 'dev' 'test' or 'predict'.
        Returns:
            a list with all the examples record.
        '''

        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a, text_pair=example.text_b, max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info('The text %s has been dropped as it has no words in the vocab after tokenization.' %
                                example.text_a)
                    continue
                if example.label:
                    record['label'] = self.label_list.index(example.label) if self.label_list else float(example.label)
                records.append(record)
                process_bar.update(1)
        return records

    def get_train_records(self, shuffle=False):
        return self.get_records('train', shuffle=shuffle)

    def get_dev_records(self, shuffle=False):
        return self.get_records('dev', shuffle=shuffle)

    def get_test_records(self, shuffle=False):
        return self.get_records('test', shuffle=shuffle)

    def get_val_records(self, shuffle=False):
        return self.get_records('val', shuffle=shuffle)

    def get_predict_records(self, shuffle=False):
        return self.get_records('predict', shuffle=shuffle)

    def get_records(self, phase, shuffle=False):
        if phase == 'train':
            records = self.train_records
        elif phase == 'dev':
            records = self.dev_records
        elif phase == 'test':
            records = self.test_records
        elif phase == 'val':
            records = self.dev_records
        elif phase == 'predict':
            records = self.predict_records
        else:
            raise ValueError('Invalid phase: %s' % phase)

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

    def batch_records_generator(self, phase, batch_size, shuffle=True, pad_to_batch_max_seq_len=False):
        ''' generate a batch of records, usually used in dynamic graph mode.
        Args:
            phase (str): the dataset phase, can be 'train', 'dev', 'val', 'test' or 'predict'.
            batch_size (int): the data batch size
            shuffle (bool): if set to True, will shuffle the dataset.
            pad_to_batch_max_seq_len (bool): if set to True, will dynamically pad to the max sequence length of the batch data.
                                             Only recommended to set to True when the model has used RNN.
        '''
        records = self.get_records(phase, shuffle=shuffle)

        batch_records = []
        batch_lens = []
        for record in records:
            batch_records.append(record)
            if pad_to_batch_max_seq_len:
                # This may reduce the processing speed
                tokens_wo_pad = [
                    token for token in self.tokenizer.decode(record, only_convert_to_tokens=True)
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
                rev_batch_records = {key: [record[key] for record in batch_records] for key in batch_records[0]}
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
            rev_batch_records = {key: [record[key] for record in batch_records] for key in batch_records[0]}
            yield rev_batch_records


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
                 split_char='\002',
                 start_token='<s>',
                 end_token='</s>',
                 unk_token='<unk>'):
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
        '''
        Returns a list[dict] including all the input information what the model need.
        Args:
            examples (list): the data example, returned by _read_file.
            phase (str): the processing phase, can be 'train' 'dev' 'test' or 'predict'.
        Returns:
            a list with all the examples record.
        '''
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a.split(self.split_char),
                    text_pair=example.text_b.split(self.split_char) if example.text_b else None,
                    max_seq_len=self.max_seq_len)
                if example.label:
                    expand_label = [self.start_token] + example.label.split(
                        self.split_char)[:self.max_seq_len - 2] + [self.end_token]
                    expand_label_id = [
                        self.label_index.get(label, self.label_index[self.unk_token]) for label in expand_label
                    ]
                    record['label'] = expand_label_id[1:] + [self.label_index[self.end_token]
                                                             ] * (self.max_seq_len - len(expand_label) + 1)
                    record['dec_input'] = expand_label_id[:-1] + [self.label_index[self.end_token]
                                                                  ] * (self.max_seq_len - len(expand_label) + 1)
                records.append(record)
                process_bar.update(1)
        return records
