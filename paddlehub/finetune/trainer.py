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

import os
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
    def __init__(self, model, strategy, use_vdl=False, checkpoint_dir=None, compare_metrics=None):
        self.nranks = ParallelEnv().nranks
        self.local_rank = ParallelEnv().local_rank
        self.model = model
        self.optimizer = strategy
        if self.nranks > 1:
            context = fluid.dygraph.prepare_context()
            self.model = fluid.dygraph.DataParallel(self.model, context)
        self.use_vdl = use_vdl
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else 'ckpt_{}'.format(time.time())
        self.epoch = 0
        self.load_checkpoint(self.checkpoint_dir)
        self.compare_metrics = self._compare_metrics if not compare_metrics else compare_metrics

    def load_checkpoint(self, checkpoint_dir: str):
        '''
        Load checkpoint and state dict from

        Args:
            checkpoint_dir(str) : Directory where checkpoints are stored
        '''
        max_epoch = -1

        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if not file.startswith('epoch_'):
                    continue

                _epoch = file.split('_')[-1]
                if not _epoch.isdigit():
                    continue

                max_epoch = max(max_epoch, int(_epoch))

        if max_epoch == -1:
            logger.warning('PaddleHub model checkpoint not found, start from scratch...')
            return

        self.epoch = max_epoch
        logger.info('PaddleHub model checkpoint loaded. current_epoch={}'.format(self.epoch))

        model_path = os.path.join(checkpoint_dir, '{}_{}'.format('epoch', self.epoch), 'model')
        state_dict, _ = fluid.load_dygraph(model_path)
        self.model.set_dict(state_dict)

    def save_checkpoint(self, checkpoint_dir: str):
        '''
        Save model checkpoint and state dict

        Args:
            checkpoint_dir(str) : Directory where checkpoints are stored
        '''
        model_path = os.path.join(checkpoint_dir, '{}_{}'.format('epoch', self.epoch), 'model')
        logger.info('Saving model checkpoint to {}'.format(model_path))

        self.save_model(model_path)

    def save_model(self, save_dir):
        '''Save model'''
        fluid.save_dygraph(self.model.state_dict(), save_dir)

    def train(self,
              train_dataset: fluid.io.Dataset,
              epochs: int = 1,
              batch_size: int = 1,
              num_workers: int = 0,
              eval_dataset: fluid.io.Dataset = None,
              log_interval: int = 10,
              save_interval: int = 10):
        '''
        Train a model with specific config.

        Args:
            train_dataset(fluid.io.Dataset) : Dataset to train the model
            epochs(int) : Number of training loops, default is 1.
            batch_size(int) : Batch size of per step, default is 1.
            num_workers(int) : Number of subprocess to load data, default is 0.
            eval_dataset(fluid.io.Dataset) : The validation dataset, deafult is None. If set, the Trainer will execute evaluate function every `save_interval` epochs.
            log_interval(int) : Log the train infomation every `log_interval` steps.
            save_interval(int) : Save the checkpoint every `save_interval` epochs.
        '''
        use_gpu = True
        place = fluid.CUDAPlace(ParallelEnv().dev_id) if use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            batch_sampler = DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)

            steps_per_epoch = len(batch_sampler)
            timer = Timer(steps_per_epoch * epochs)
            timer.start()
            best_score = None

            for i in range(epochs):
                self.epoch += 1
                avg_loss = 0
                avg_metrics = defaultdict(int)
                self.model.train()

                for batch_idx, batch in enumerate(loader):
                    loss, metrics = self.training_step(batch, batch_idx)
                    self.optimizer_step(self.epoch, batch_idx, self.optimizer, loss)
                    self.optimizer_zero_grad(self.epoch, batch_idx, self.optimizer)

                    # calculate metrics and loss
                    avg_loss += loss.numpy()[0]
                    for metric, value in metrics.items():
                        avg_metrics[metric] += value.numpy()[0]

                    timer.step()

                    if (batch_idx + 1) % log_interval == 0 and self.local_rank == 0:
                        lr = self.optimizer.current_step_lr()
                        avg_loss /= log_interval

                        print_msg = 'Epoch={}/{}, Step={}/{}'.format(self.epoch, epochs, batch_idx + 1, steps_per_epoch)
                        print_msg += ' loss={:.4f}'.format(avg_loss)

                        for metric, value in avg_metrics.items():
                            print_msg += ' {}={:.4f}'.format(metric, value / log_interval)

                        print_msg += ' lr={:.6f}'.format(lr)

                        logger.train(print_msg)

                        avg_loss = 0
                        avg_metrics = defaultdict(int)

                    if self.epoch % save_interval == 0 and batch_idx + 1 == steps_per_epoch and self.local_rank == 0:
                        if eval_dataset:
                            result = self.evaluate(eval_dataset, batch_size, num_workers)['metrics']
                            if not best_score or self.compare_metrics(best_score, result):
                                best_score = result['metrics']
                                best_model_path = os.path.join(self.checkpoint_dir, 'best_model')
                                self.save_model(best_model_path)

                                logger.info('Saving best model to {}'.format(best_model_path))

                        self.save_checkpoint(self.checkpoint_dir)

    def evaluate(self, eval_dataset: fluid.io.Dataset, batch_size: int = 1, num_workers: int = 0):
        '''
        Run evaluation and returns metrics.

        Args:
            eval_dataset(fluid.io.Dataset) : The validation dataset
            batch_size(int) : Batch size of per step, default is 1.
            num_workers(int) : Number of subprocess to load data, default is 0.
        '''
        use_gpu = True
        place = fluid.CUDAPlace(ParallelEnv().dev_id) if use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            batch_sampler = DistributedBatchSampler(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            loader = DataLoader(
                eval_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)

            self.model.eval()
            steps = len(batch_sampler)
            timer = Timer(steps)
            timer.start()
            avg_loss = num_samples = 0
            avg_metrics = defaultdict(int)

            for batch_idx, batch in enumerate(loader):
                result = self.validation_step(batch, batch_idx)
                loss = result.get('loss', None)
                metrics = result.get('metrics', {})
                bs = batch[0].shape[0]
                num_samples += bs

                if loss:
                    avg_loss += loss.numpy()[0] * bs

                for metric, value in metrics.items():
                    avg_metrics[metric] += value.numpy()[0] * bs

            # print avg metrics and loss
            print_msg = '[Evaluation result]'
            if loss:
                print_msg += ' avg_loss={:.4f}'.format(avg_loss / num_samples)

            for metric, value in avg_metrics.items():
                print_msg += ' avg_{}={:.4f}'.format(metric, value / num_samples)

            logger.eval(print_msg)

            if loss:
                return {'loss': avg_loss, 'metrics': avg_metrics}
            return {'metrics': avg_metrics}

    def training_step(self, batch, batch_idx):
        result = self.model.training_step(batch, batch_idx)

        # process result
        if not isinstance(result, dict):
            raise RuntimeError()

        loss = result.get('loss', None)
        if not loss:
            raise RuntimeError()

        metrics = result.get('metrics', {})

        # back prop
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

    def _compare_metrics(self, old_metric, new_metric):
        '''Compare the whether the new metric value is better than the old one'''
        mainkey = list(new_metric.keys())[0]
        return old_metric[mainkey] < new_metric[mainkey]
