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
                 predict_file_with_header=False):
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
