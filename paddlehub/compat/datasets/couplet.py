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

import codecs
import csv
import os

import paddlehub.env as hubenv
from paddlehub.compat.datasets.base_dataset import InputExample
from paddlehub.compat.datasets.nlp_dataset import GenerationDataset
from paddlehub.utils.download import download_data


@download_data('https://bj.bcebos.com/paddlehub-dataset/couplet.tar.gz')
class Couplet(GenerationDataset):
    '''An open source couplet dataset, see https://github.com/v-zich/couplet-clean-dataset for details.'''

    def __init__(self, tokenizer=None, max_seq_len=None):
        dataset_dir = os.path.join(hubenv.DATA_HOME, 'couplet')
        with open(os.path.join(dataset_dir, 'vocab.txt'), encoding='utf8') as vocab_file:
            label_list = [line.strip() for line in vocab_file.readlines()]
        super(Couplet, self).__init__(
            base_path=dataset_dir,
            train_file='train.tsv',
            dev_file='dev.tsv',
            test_file='test.tsv',
            label_list=label_list,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len)

    def _read_file(self, input_file, phase=None):
        '''Reads a tab separated value file.'''
        with codecs.open(input_file, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=None)
            examples = []
            seq_id = 0
            for line in reader:
                example = InputExample(guid=seq_id, label=line[1], text_a=line[0])
                seq_id += 1
                examples.append(example)

            return examples
