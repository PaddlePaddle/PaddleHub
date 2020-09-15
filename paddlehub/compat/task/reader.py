# coding:utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import namedtuple
from typing import Callable, Generator, Generic, List

import numpy as np

from paddlehub.utils.log import logger
from paddlehub.compat.task import tokenization
from paddlehub.compat.task.batch import pad_batch_data


class InputExample(object):
    '''
    Input data structure of BERT/ERNIE, can satisfy single sequence task like
    text classification, sequence lableing; Sequence pair task like dialog
    task.
    '''

    def __init__(self, guid: int, text_a: str, text_b: str = None, label: str = None):
        '''Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    '''
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b is None:
            return 'text={}\tlabel={}'.format(self.text_a, self.label)
        else:
            return 'text_a={}\ttext_b={},label={}'.format(self.text_a, self.text_b, self.label)


class BaseReader(object):
    def __init__(self, dataset: Generic, random_seed: int = None):
        self.dataset = dataset
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        np.random.seed(random_seed)

        # generate label map
        self.label_map = {}
        try:
            for index, label in enumerate(self.dataset.get_labels()):
                self.label_map[label] = index
            logger.info('Dataset label map = {}'.format(self.label_map))
        except:
            # some dataset like squad, its label_list=None
            logger.info('Dataset is None or it has not any labels, label map = {}'.format(self.label_map))

    def get_train_examples(self) -> List:
        return self.dataset.get_train_examples()

    def get_dev_examples(self) -> List:
        return self.dataset.get_dev_examples()

    def get_test_examples(self) -> List:
        return self.dataset.get_test_examples()

    def data_generator(self) -> Generic:
        raise NotImplementedError


class BaseNLPReader(BaseReader):
    def __init__(self,
                 vocab_path: str,
                 dataset: Generic = None,
                 max_seq_len: int = 512,
                 do_lower_case: bool = True,
                 random_seed: int = None,
                 sp_model_path: str = None,
                 word_dict_path: str = None,
                 in_tokens: bool = False):
        super(BaseNLPReader, self).__init__(dataset, random_seed)
        self.max_seq_len = max_seq_len
        if sp_model_path and word_dict_path:
            self.tokenizer = tokenization.WSSPTokenizer(vocab_path, sp_model_path, word_dict_path, ws=True, lower=True)
        else:
            self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab['[PAD]']
        self.cls_id = self.vocab['[CLS]']
        self.sep_id = self.vocab['[SEP]']
        self.mask_id = self.vocab['[MASK]']
        self.in_tokens = in_tokens

        self.Record_With_Label_Id = namedtuple('Record', ['token_ids', 'text_type_ids', 'position_ids', 'label_id'])
        self.Record_Wo_Label_Id = namedtuple('Record', ['token_ids', 'text_type_ids', 'position_ids'])

    def _truncate_seq_pair(self, tokens_a: List, tokens_b: List, max_length: int):
        '''Truncates a sequence pair in place to the maximum length.'''

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
                                   example: InputExample,
                                   max_seq_length: int,
                                   tokenizer: Generic,
                                   phase: str = None) -> namedtuple:
        '''Converts a single `Example` into a single `Record`.'''

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if example.text_b is not None:
            #if 'text_b' in example._fields:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with '- 3'
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with '- 2'
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
        # Where 'type_ids' are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the 'sentence vector'. Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        text_type_ids = []
        tokens.append('[CLS]')
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append('[SEP]')
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append('[SEP]')
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))

        if self.label_map:
            if example.label not in self.label_map:
                raise KeyError('example.label = {{{}}} not in label'.format(example.label))
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        if phase != 'predict':
            record = self.Record_With_Label_Id(
                token_ids=token_ids, text_type_ids=text_type_ids, position_ids=position_ids, label_id=label_id)
        else:
            record = self.Record_Wo_Label_Id(
                token_ids=token_ids, text_type_ids=text_type_ids, position_ids=position_ids)

        return record

    def _pad_batch_records(self, batch_records: List, phase: str):
        raise NotImplementedError

    def _prepare_batch_data(self, examples: List, batch_size: int, phase: str = None) -> Generator:
        '''generate batch records'''
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == 'train':
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer, phase)
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
                       batch_size: int = 1,
                       phase: str = 'train',
                       shuffle: bool = True,
                       data: List = None,
                       return_list: bool = True) -> Callable:
        if phase != 'predict' and not self.dataset:
            raise ValueError('The dataset is None ! It isn\'t allowed.')
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
                    item_i = InputExample(guid=seq_id, text_a=item[0], label=label)
                elif len(item) == 2:
                    item_i = InputExample(guid=seq_id, text_a=item[0], text_b=item[1], label=label)
                else:
                    raise ValueError('The length of input_text is out of handling, which must be 1 or 2!')
                examples.append(item_i)
                seq_id += 1
        else:
            raise ValueError('Unknown phase, which should be in [\'train\', \'dev\', \'test\', \'predict\'].')

        def wrapper():
            if shuffle:
                np.random.shuffle(examples)

            for batch_data in self._prepare_batch_data(examples, batch_size, phase=phase):
                if return_list:
                    # for DataFeeder
                    yield [batch_data]
                else:
                    # for DataLoader
                    yield batch_data

        return wrapper


class ClassifyReader(BaseNLPReader):
    def _pad_batch_records(self, batch_records: List, phase: str = None) -> List:
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(batch_text_type_ids, max_seq_len=self.max_seq_len, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(batch_position_ids, max_seq_len=self.max_seq_len, pad_idx=self.pad_id)

        return_list = [padded_token_ids, padded_position_ids, padded_text_type_ids, input_mask, batch_seq_lens]
        if phase != 'predict':
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype('int64').reshape([-1, 1])
            return_list += [batch_labels]

        return return_list
