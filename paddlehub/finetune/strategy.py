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
import multiprocessing

import paddle.fluid as fluid

from paddlehub.finetune.optimization import adam_weight_decay_optimization
from paddlehub.finetune.regularizer import L2SPDecayRegularizer


def get_pretrained_parameter(main_program, start_program):
    pretrained_parameters = []
    global_block = main_program.global_block()
    for op in global_block.ops[::-1]:
        for input_arg in op.input_arg_names:
            var = global_block.var(input_arg)
            if isinstance(
                    var, fluid.framework.Parameter
            ) and input_arg not in start_program.global_block().vars:
                pretrained_parameters.append(var)

    return pretrained_parameters


class DefaultStrategy(object):
    def __init__(self, learning_rate=1e-4, optimizer_name="adam"):
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        if self._optimizer_name.lower() == "sgd":
            self.optimizer = fluid.optimizer.SGD(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "adagrad":
            self.optimizer = fluid.optimizer.Adagrad(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "adamax":
            self.optimizer = fluid.optimizer.Adamax(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "decayedadagrad":
            self.optimizer = fluid.optimizer.DecayedAdagrad(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "ftrl":
            self.optimizer = fluid.optimizer.Ftrl(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "larsmomentum":
            self.optimizer = fluid.optimizer.LarsMomentum(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "momentum":
            self.optimizer = fluid.optimizer.Momentum(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "decayedadagrad":
            self.optimizer = fluid.optimizer.DecayedAdagrad(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "rmsprop":
            self.optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=self.learning_rate)
        else:
            self.optimizer = fluid.optimizer.Adam(
                learning_rate=self.learning_rate)

    def execute(self, loss, data_reader, config):
        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultStrategy's optimizer is None")

    # TODO complete __str__()
    def __str__(self):
        return "DefaultStrategy"


class AdamWeightDecayStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 lr_scheduler="linear_decay",
                 warmup_proportion=0.1,
                 weight_decay=0.01,
                 optimizer_name="adam"):
        super(AdamWeightDecayStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        # check strategy correctness
        if lr_scheduler not in ["linear_decay", "noam_decay"]:
            raise ValueError("lr_scheduler {} is not setup "
                             "correctly".format(lr_scheduler))
        self._lr_scheduler = lr_scheduler
        self._warmup_proportion = warmup_proportion
        self._weight_decay = weight_decay

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def warmup_proportion(self):
        return self._warmup_proportion

    @property
    def weight_decay(self):
        return self._weight_decay

    def execute(self, loss, data_reader, config):
        main_program = loss.block.program
        # calculate wamrup step
        dev_count = self._get_dev_count(config)
        data_reader.data_generator(
            batch_size=config.batch_size, phase='train', shuffle=True)
        data_reader.data_generator(
            batch_size=config.batch_size, phase='val', shuffle=False)
        data_reader.data_generator(
            batch_size=config.batch_size, phase='dev', shuffle=False)
        num_train_examples = data_reader.get_num_examples(phase='train')
        max_train_steps = config.num_epoch * num_train_examples // config.batch_size // dev_count
        warmup_steps = int(max_train_steps * self.warmup_proportion)

        scheduled_lr = adam_weight_decay_optimization(
            loss, warmup_steps, max_train_steps, self.learning_rate,
            main_program, self.weight_decay, self.lr_scheduler)

        return scheduled_lr

    def _get_dev_count(self, config):
        if config.use_cuda:
            dev_count = fluid.core.get_cuda_device_count()
        else:
            dev_count = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        return dev_count

    # TODO complete __str__()
    def __str__(self):
        return "AdamWeightDecayStrategy"


class DefaultFinetuneStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 regularization_coeff=1e-3):
        super(DefaultFinetuneStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        self.regularization_coeff = regularization_coeff

    def execute(self, loss, data_reader, config):
        # get pretrained parameters
        program = loss.block.program
        global_block = program.global_block()
        pretrained_params = get_pretrained_parameter(
            program, fluid.default_startup_program())

        # set parameter attrs
        for index, param in enumerate(pretrained_params):
            param.regularizer = fluid.regularizer.L2Decay(
                regularization_coeff=self.regularization_coeff)

        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultFinetuneStrategy's optimizer is None")


class L2SPFinetuneStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 regularization_coeff=1e-3):
        super(L2SPFinetuneStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        self.regularization_coeff = regularization_coeff

    def execute(self, loss, data_reader, config):
        # get pretrained parameters
        program = loss.block.program
        global_block = program.global_block()
        pretrained_params = get_pretrained_parameter(
            program, fluid.default_startup_program())

        # set parameter attrs
        for index, param in enumerate(pretrained_params):
            param.regularizer = L2SPDecayRegularizer(
                regularization_coeff=self.regularization_coeff)

        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultFinetuneStrategy's optimizer is None")
