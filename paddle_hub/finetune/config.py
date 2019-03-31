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

import collections


class FinetuneConfig(object):
    """ This class specifies the configurations for PaddleHub to finetune """

    def __init__(self,
                 log_interval=10,
                 eval_interval=100,
                 save_ckpt_interval=None,
                 use_cuda=False,
                 learning_rate=1e-4,
                 checkpoint_dir=None,
                 num_epoch=10,
                 batch_size=None,
                 max_seq_len=128,
                 weight_decay=None,
                 warmup_proportion=0.0,
                 finetune_strategy=None,
                 enable_memory_optim=True,
                 optimizer="adam"):
        """ Construct finetune Config """
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._save_ckpt_interval = save_ckpt_interval
        self._use_cuda = use_cuda
        self._learning_rate = learning_rate
        self._checkpoint_dir = checkpoint_dir
        self._num_epoch = num_epoch
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._weight_decay = weight_decay
        self._warmup_proportion = warmup_proportion
        self._finetune_strategy = finetune_strategy
        self._enable_memory_optim = enable_memory_optim
        self._optimizer = optimizer

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
    def learning_rate(self):
        return self._learning_rate

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
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def weight_decay(self):
        return self._weight_decay

    @property
    def warmup_proportion(self):
        return self._warmup_proportion

    @property
    def finetune_strategy(self):
        return self._finetune_strategy

    @property
    def enable_memory_optim(self):
        return self._enable_memory_optim

    @property
    def optimier(self):
        return self._optimizer
