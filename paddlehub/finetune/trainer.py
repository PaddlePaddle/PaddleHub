# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import DataLoader
from paddle.incubate.hapi.distributed import DistributedBatchSampler

from paddlehub.utils.log import logger
from paddlehub.utils.utils import Timer


class Trainer(object):
    def __init__(self, model, strategy, use_vdl=False, ckpt_dir=None):
        self.nranks = ParallelEnv().nranks
        self.local_rank = ParallelEnv().local_rank
        self.model = model
        self.optimizer = strategy
        if self.nranks > 1:
            context = fluid.dygraph.prepare_context()
            self.model = fluid.dygraph.DataParallel(self.model, context)
        self.use_vdl = use_vdl
        self.ckpt_dir = ckpt_dir if ckpt_dir else 'ckpt_{}'.format(time.time())

    def train(self,
              train_dataset,
              epochs=1,
              batch_size=1,
              num_workers=1,
              eval_dataset=None,
              log_interval=10,
              save_ckpt_interval=10):
        use_gpu = False
        place = fluid.CUDAPlace(ParallelEnv().dev_id) if use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            batch_sampler = DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)

            self.model.train()
            steps_per_epoch = len(batch_sampler)
            timer = Timer(steps_per_epoch * epochs)
            timer.start()

            for current_epoch in range(epochs):
                avg_loss = 0
                avg_metrics = defaultdict(int)
                for batch_idx, batch in enumerate(loader):
                    loss, metrics = self.training_step(batch, batch_idx)
                    self.optimizer_step(current_epoch, batch_idx, self.optimizer, loss)
                    self.optimizer_zero_grad(current_epoch, batch_idx, self.optimizer)

                    # calculate metrics and loss
                    avg_loss += loss.numpy()[0]
                    for metric, value in metrics.items():
                        avg_metrics[metric] += value.numpy()[0]

                    timer.step()

                    if (batch_idx + 1) % log_interval == 0 and self.local_rank == 0:
                        lr = self.optimizer.current_step_lr()
                        avg_loss /= log_interval
                        for metric, value in avg_metrics.items():
                            value /= log_interval

                        print_msg = 'Epoch={}/{}, Step={}/{}'.format(current_epoch + 1, epochs, batch_idx + 1,
                                                                     steps_per_epoch)
                        print_msg += ' loss={:.4f}'.format(avg_loss)

                        for metric, value in avg_metrics.items():
                            print_msg += ' {}={:.4f}'.format(metric, value)

                        print_msg += ' lr={:.6f}'.format(lr)

                        logger.train(print_msg)

                        avg_loss = 0
                        avg_metrics = defaultdict(int)

                    if (batch_idx + 1) % save_ckpt_interval == 0 and self.local_rank == 0:
                        if eval_dataset:
                            self.evaluate(eval_dataset, batch_size, num_workers)

    def training_step(self, batch, batch_idx):
        result = self.model.training_step(batch, batch_idx)

        # process result
        if not isinstance(result, dict):
            raise RuntimeError()

        loss = result.get('loss', None)
        if not loss:
            raise RuntimeError()

        metrics = result.get('metrics', {})

        # backprop
        if self.nranks > 1:
            self.model.scale_loss(loss)
            loss.backward()
            self.model.apply_collective_grads()
        else:
            loss.backward()

        return loss, metrics

    def validation_step(self, batch, batch_idx):
        result = self.model.validation_step(batch, batch_idx)
        return result

    def optimizer_step(self, current_epoch, batch_idx, optimizer, loss):
        self.optimizer.minimize(loss)

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer):
        self.model.clear_gradients()

    def is_better_score(self, old_score, new_score):
        if hasattr(self.model, 'is_better_score'):
            return self.model.is_better_score(old_score, new_score)

        mainkey = list(new_score.keys())[0]
        return old_score[mainkey] < new_score[mainkey]

    def evaluate(self, eval_dataset, batch_size=1, num_workers=1):
        use_gpu = False
        place = fluid.CUDAPlace(ParallelEnv().dev_id) if use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            batch_sampler = DistributedBatchSampler(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            loader = DataLoader(
                eval_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)

            self.model.eval()
            steps = len(batch_sampler)
            timer = Timer(steps)
            timer.start()

            for batch_idx, batch in enumerate(loader):
                result = self.validation_step(batch, batch_idx)
                print_msg = 'Step={}/{}'.format(batch_idx, steps)

                for metric, value in result.items():
                    print_msg += ' {}={:.4f}'.format(metric, value)

                logger.eval(print_msg)
