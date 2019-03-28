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

FinetuneConfig = collections.namedtuple(
    'FinetuneConfig',
    [
        'log_interval',  # print training log every n step
        'eval_interval',  # evalution the model every n steps
        'save_ckpt_interval',  # save the model checkpoint every n steps
        'use_cuda',  # use gpu or not
        'learning_rate',
        'checkpoint_dir',  # model checkpoint directory
        'num_epoch',  # number of finetune epoch
        'batch_size',
        # for bert parameter
        'max_seq_len',  # for bert
        'weight_decay',  # for bert
        'warmup_proportion',  # for bert
        'in_tokens',  # for bert
        'strategy',
        'with_memory_optimization'
    ])
