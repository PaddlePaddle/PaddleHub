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

import time
from typing import Callable


class RunConfig(object):
    ''' This class specifies the configurations for PaddleHub to finetune '''

    def __init__(self,
                 log_interval: int = 10,
                 eval_interval: int = 100,
                 use_data_parallel: bool = True,
                 save_ckpt_interval: int = None,
                 use_cuda: bool = True,
                 checkpoint_dir: str = None,
                 num_epoch: int = 1,
                 batch_size: int = 32,
                 strategy: Callable = None):
        ''' Construct finetune Config '''
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_ckpt_interval = save_ckpt_interval
        self.use_cuda = use_cuda
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.use_data_parallel = use_data_parallel

        if checkpoint_dir is None:
            now = int(time.time())
            time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(now))
            self.checkpoint_dir = 'ckpt_' + time_str
        else:
            self.checkpoint_dir = checkpoint_dir

    def __repr__(self):
        return 'config with num_epoch={}, batch_size={}, use_cuda={}, checkpoint_dir={} '.format(
            self.num_epoch, self.batch_size, self.use_cuda, self.checkpoint_dir)
