# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import os

from paddlehub.utils.log import logger


class InputExample(object):
    '''
    Input data structure of BERT/ERNIE, can satisfy single sequence task like
    text classification, sequence lableing; Sequence pair task like dialog
    task.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    '''

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b is None:
            return 'text={}\tlabel={}'.format(self.text_a, self.label)
        else:
            return 'text_a={}\ttext_b={},label={}'.format(self.text_a, self.text_b, self.label)


class BaseDataset(object):
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
                 predict_file_with_header=False):
        if not (train_file or dev_file or test_file):
            raise ValueError('At least one file should be assigned')
        self.base_path = base_path
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.predict_file = predict_file
        self.label_file = label_file
        self.label_list = label_list

        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []
        self.predict_examples = []

        self.if_file_with_header = {
            'train': train_file_with_header,
            'dev': dev_file_with_header,
            'test': test_file_with_header,
            'predict': predict_file_with_header
        }

        if train_file:
            self._load_train_examples()
        if dev_file:
            self._load_dev_examples()
        if test_file:
            self._load_test_examples()
        if predict_file:
            self._load_predict_examples()
        if self.label_file:
            if not self.label_list:
                self.label_list = self._load_label_data()
            else:
                logger.warning('As label_list has been assigned, label_file is noneffective')

        if self.label_list:
            self.label_index = dict(zip(self.label_list, range(len(self.label_list))))

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_val_examples(self):
        return self.get_dev_examples()

    def get_predict_examples(self):
        return self.predict_examples

    def get_examples(self, phase):
        if phase == 'train':
            return self.get_train_examples()
        elif phase == 'dev':
            return self.get_dev_examples()
        elif phase == 'test':
            return self.get_test_examples()
        elif phase == 'val':
            return self.get_val_examples()
        elif phase == 'predict':
            return self.get_predict_examples()
        else:
            raise ValueError('Invalid phase: %s' % phase)

    def get_labels(self):
        return self.label_list

    @property
    def num_labels(self):
        return len(self.label_list)

    # To be compatible with ImageClassificationDataset
    def label_dict(self):
        return {index: key for index, key in enumerate(self.label_list)}

    def _load_train_examples(self):
        self.train_path = os.path.join(self.base_path, self.train_file)
        self.train_examples = self._read_file(self.train_path, phase='train')

    def _load_dev_examples(self):
        self.dev_path = os.path.join(self.base_path, self.dev_file)
        self.dev_examples = self._read_file(self.dev_path, phase='dev')

    def _load_test_examples(self):
        self.test_path = os.path.join(self.base_path, self.test_file)
        self.test_examples = self._read_file(self.test_path, phase='test')

    def _load_predict_examples(self):
        self.predict_path = os.path.join(self.base_path, self.predict_file)
        self.predict_examples = self._read_file(self.predict_path, phase='predict')

    def _read_file(self, path, phase=None):
        raise NotImplementedError

    def _load_label_data(self):
        with open(os.path.join(self.base_path, self.label_file), 'r', encoding='utf8') as file:
            return file.read().strip().split('\n')

    def __str__(self):
        return 'Dataset: %s with %i train examples, %i dev examples and %i test examples' % (
            self.__class__.__name__, len(self.train_examples), len(self.dev_examples), len(self.test_examples))
