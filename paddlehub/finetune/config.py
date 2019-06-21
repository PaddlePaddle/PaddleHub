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

from datetime import datetime
import time

from paddlehub.finetune.strategy import DefaultStrategy
from paddlehub.common.logger import logger


class RunConfig(object):
    """ This class specifies the configurations for PaddleHub to finetune """

    def __init__(self,
                 log_interval=10,
                 eval_interval=100,
                 use_pyreader=False,
                 use_data_parallel=False,
                 save_ckpt_interval=None,
                 use_cuda=True,
                 checkpoint_dir=None,
                 num_epoch=1,
                 batch_size=32,
                 enable_memory_optim=True,
                 strategy=None):
        """ Construct finetune Config """
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._save_ckpt_interval = save_ckpt_interval
        self._use_cuda = use_cuda
        self._checkpoint_dir = checkpoint_dir
        self._num_epoch = num_epoch
        self._batch_size = batch_size
        self._use_pyreader = use_pyreader
        self._use_data_parallel = use_data_parallel
        if strategy is None:
            self._strategy = DefaultStrategy()
        else:
            self._strategy = strategy
        self._enable_memory_optim = enable_memory_optim
        if checkpoint_dir is None:

            now = int(time.time())
            time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
            self._checkpoint_dir = "ckpt_" + time_str
        else:
            self._checkpoint_dir = checkpoint_dir
        logger.info("Checkpoint dir: {}".format(self._checkpoint_dir))

    @property
    def log_interval(self):
        return self._log_interval

    @property
    def eval_interval(self):
        return self._eval_interval

    @property
    def save_ckpt_interval(self):
        return self._save_ckpt_interval

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def num_epoch(self):
        return self._num_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def strategy(self):
        return self._strategy

    @property
    def enable_memory_optim(self):
        return self._enable_memory_optim

    @property
    def use_pyreader(self):
        return self._use_pyreader

    @property
    def use_data_parallel(self):
        return self._use_data_parallel
