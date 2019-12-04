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

import paddlehub as hub
from paddlehub.common.downloader import default_downloader
from paddlehub.common.logger import logger


class InputExample(object):
    """
    Input data structure of BERT/ERNIE, can satisfy single sequence task like
    text classification, sequence lableing; Sequence pair task like dialog
    task.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b is None:
            return "text={}\tlabel={}".format(self.text_a, self.label)
        else:
            return "text_a={}\ttext_b={},label={}".format(
                self.text_a, self.text_b, self.label)


class HubDataset(object):
    def __init__(self,
                 base_path,
                 train_file=None,
                 dev_file=None,
                 test_file=None,
                 label_file=None,
                 label_list=None,
                 init_phase="train"):
        if not (train_file or dev_file or test_file):
            raise ValueError("At least one file should be assigned")
        self.base_path = base_path
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.label_file = label_file
        self.label_list = label_list

        if init_phase not in ["train", "dev", "val", "test"]:
            raise ValueError("phase only support train/dev/val/test")
        if init_phase == "val":
            init_phase = "dev"
        self._current_phase = init_phase

        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

        self._load_train_examples()
        self._load_dev_examples()
        self._load_test_examples()

        if self.label_file:
            if not self.label_list:
                self.label_list = self._load_label_data()
            else:
                logger.warning(
                    "As label_list has been assigned, label_file will be disabled"
                )

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_val_examples(self):
        return self.get_dev_examples()

    def get_labels(self):
        return self.label_list

    @property
    def num_labels(self):
        return len(self.label_list)

    def label_dict(self):
        return {index: key for index, key in enumerate(self.label_list)}

    def _download_dataset(self, dataset_path, url):
        if not os.path.exists(dataset_path):
            result, tips, dataset_path = default_downloader.download_file_and_uncompress(
                url=url,
                save_path=hub.common.dir.DATA_HOME,
                print_progress=True,
                replace=True)
            if not result:
                raise Exception(tips)
        else:
            logger.info("Dataset {} already cached.".format(dataset_path))
        return dataset_path

    def _load_train_examples(self):
        self.train_file = os.path.join(self.base_path, self.train_file)
        self.train_examples = self._read_file(self.train_file, phase="train")

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.base_path, self.dev_file)
        self.dev_examples = self._read_file(self.dev_file, phase="dev")

    def _load_test_examples(self):
        self.test_file = os.path.join(self.base_path, self.test_file)
        self.test_examples = self._read_file(self.test_file, phase="test")

    def _read_file(self, path, phase=None):
        raise NotImplementedError

    def _load_label_data(self):
        with open(os.path.join(self.base_path, self.label_file), "r") as file:
            return file.read().split("\n")

    def set_current_phase(self, phase):
        if phase not in ["train", "dev", "val", "test"]:
            raise ValueError("phase only support train/dev/val/test")
        if phase == "val":
            phase = "dev"
        self._current_phase = phase

    @property
    def current_phase(self):
        return self._current_phase

    def __len__(self):
        return len(eval("self.%s_examples" % self.current_phase))

    def __getitem__(self, item):
        return eval("self.%s_examples[%s]" % (self.current_phase, item))

    def __iter__(self):
        return eval("self.%s_examples" % self.current_phase)

    def __str__(self):
        return "Dataset: %s with %i train examples, %i dev examples and %i test examples" % (
            self.__class__.__name__, len(self.train_examples),
            len(self.dev_examples), len(self.test_examples))
