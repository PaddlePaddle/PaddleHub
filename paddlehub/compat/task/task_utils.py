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

import copy
import time
from typing import Any

import paddle


class RunState(object):
    '''
    RunState is used to save the result of every running step
    Args:
        length (int): the number of fetch result
    '''

    def __init__(self, length: int):
        self.run_time_begin = time.time()
        self.run_step = 0
        self.run_examples = 0
        self.run_results = [0] * length
        self.run_time_used = 0
        self.run_speed = 0.0

    def __add__(self, other):
        self.run_step += other.run_step
        self.run_examples += other.run_examples
        for index in range(len(self.run_results)):
            self.run_results[index] += other.run_results[index]
        return self

    def update(self):
        self.run_time_used = time.time() - self.run_time_begin
        self.run_speed = self.run_step / self.run_time_used
        return self


class RunEnv(object):
    '''RunEnv saves the running environment of the train/dev/predict phase, including program, reader, metrics and so on.'''

    def __init__(self):
        self.current_epoch = 0
        self.current_step = 0
        self.main_program = None
        self.start_program = None
        self.main_program_compiled = None
        self.py_reader = None
        self.generator = None
        self.loss = None
        self.labels = None
        self.metrics = None
        self.is_inititalized = False
        self.UNG = copy.deepcopy(paddle.fluid.unique_name.generator)

    def __setattr__(self, key: str, value: Any):
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        return self.__dict__[key]
