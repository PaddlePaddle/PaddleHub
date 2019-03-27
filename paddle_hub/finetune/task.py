#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import collections
import paddle.fluid as fluid
import time
import numpy as np
import multiprocessing

from paddle_hub.finetune.optimization import bert_optimization
from paddle_hub.finetune.config import FinetuneConfig


class Task(object):
    def __init__(self, task_type, graph_var_dict, main_program,
                 startup_program):
        self.task_type = task_type
        self.graph_var_dict = graph_var_dict
        self._main_program = main_program
        self._startup_program = startup_program

    def variable(self, var_name):
        if var_name in self.graph_var_dict:
            return self.graph_var_dict[var_name]

        raise KeyError("var_name {} not in task graph".format(var_name))

    def main_program(self):
        return self._main_program

    def startup_program(self):
        return self._startup_program
