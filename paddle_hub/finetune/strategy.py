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

import os
import multiprocessing
import paddle.fluid as fluid

from .optimization import adam_weight_decay_optimization


class DefaultStrategy(object):
    def __init__(self, learning_rate=1e-4, optimizer_name="adam"):
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name

    def execute(self, loss):
        if self.optimizer.lower() == "adam":
            self.optimizer = fluid.optimizer.Adam(
                learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            self.optimizer = fluid.optimizer.SGD(
                learning_rate=self.learning_rate)

        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultStrategy's optimizer is None")


class BERTFinetuneStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 warmup_strategy="linear_warmup_decay",
                 warmup_proportion=0.0,
                 weight_decay=0.01,
                 optimizer_name=None):
        super().__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        # check strategy correctness
        if warmup_strategy not in ["linear_warmup_decay", "noam_decay"]:
            raise ValueError("warmup strategy {} is not setup "
                             "correctly".format(warmup_strategy))
        self._warmup_strategy = warmup_strategy
        self._warmup_proportion = warmup_proportion
        self._weight_decay = weight_decay

    @property
    def warmup_strategy(self):
        return self._warmup_strategy

    @property
    def warmup_proportion(self):
        return self._warmup_proportion

    @property
    def weight_decay(self):
        return self._weight_decay

    def execute(self, loss, main_program, data_reader, config):
        # calculate wamrup step
        dev_count = self._get_dev_count(config)
        num_train_examples = data_reader.get_num_examples(phase='train')
        max_train_steps = config.num_epoch * num_train_examples // config.batch_size // dev_count
        warmup_steps = int(max_train_steps * self.warmup_proportion)

        scheduled_lr = adam_weight_decay_optimization(
            loss, warmup_steps, max_train_steps, self.learning_rate,
            main_program, self.weight_decay, self.warmup_strategy)

        return scheduled_lr

    def _get_dev_count(self, config):
        if config.use_cuda:
            dev_count = fluid.core.get_cuda_device_count()
        else:
            dev_count = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        return dev_count
