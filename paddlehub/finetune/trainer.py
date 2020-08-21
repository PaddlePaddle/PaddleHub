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
import pickle
import time
from collections import defaultdict
from typing import Any, Callable

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import DataLoader
from paddle.incubate.hapi.distributed import DistributedBatchSampler
from visualdl import LogWriter

from paddlehub.utils.log import logger
from paddlehub.utils.utils import Timer


class Trainer(object):
    '''
    Trainer
    '''

    def __init__(self,
                 model: fluid.dygraph.Layer,
                 strategy: fluid.optimizer.Optimizer,
                 use_vdl: bool = True,
                 checkpoint_dir: str = None,
                 compare_metrics: Callable = None):
        self.nranks = ParallelEnv().nranks
        self.local_rank = ParallelEnv().local_rank
        self.model = model
        self.optimizer = strategy
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else 'ckpt_{}'.format(time.time())

        if self.local_rank == 0 and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.use_vdl = use_vdl
        if self.local_rank == 0 and self.use_vdl:
            vdl_dir = os.path.join(self.checkpoint_dir, 'visualization')
            self.log_writer = LogWriter(vdl_dir)

        self.current_epoch = 0
        self.best_metrics = defaultdict(int)

        if self.nranks > 1:
            context = fluid.dygraph.prepare_context()
            self.model = fluid.dygraph.DataParallel(self.model, context)
        self.compare_metrics = self._compare_metrics if not compare_metrics else compare_metrics

        self._load_checkpoint()

    def _load_checkpoint(self):
        '''Load checkpoint and state dict'''
        max_epoch = -1

        for file in os.listdir(self.checkpoint_dir):
            if not file.startswith('epoch_'):
                continue

            _epoch = file.split('_')[-1]
            if not _epoch.isdigit():
                continue

            max_epoch = max(max_epoch, int(_epoch))

        if max_epoch == -1:
            if self.local_rank == 0:
                logger.warning('PaddleHub model checkpoint not found, start from scratch...')
            return

        # load best metrics
        self._load_metrics()

        self.current_epoch = max_epoch
        metric_msg = ['{}={:.4f}'.format(metric, value) for metric, value in self.best_metrics.items()]
        metric_msg = ' '.join(metric_msg)
        if self.local_rank == 0:
            logger.info('PaddleHub model checkpoint loaded. current_epoch={} [{}]'.format(
                self.current_epoch, metric_msg))

        # load model from checkpoint
        model_path = os.path.join(self.checkpoint_dir, '{}_{}'.format('epoch', self.current_epoch), 'model')
        state_dict, _ = fluid.load_dygraph(model_path)
        self.model.set_dict(state_dict)

    def _save_checkpoint(self):
        '''Save model checkpoint and state dict'''
        model_path = os.path.join(self.checkpoint_dir, '{}_{}'.format('epoch', self.current_epoch), 'model')
        logger.info('Saving model checkpoint to {}'.format(model_path))
        self.save_model(model_path)

    def save_model(self, save_dir: str):
        '''Save model'''
        fluid.save_dygraph(self.model.state_dict(), save_dir)

    def _save_metrics(self):
        with open(os.path.join(self.checkpoint_dir, 'metrics.pkl'), 'wb') as file:
            pickle.dump(self.best_metrics, file)

    def _load_metrics(self):
        with open(os.path.join(self.checkpoint_dir, 'metrics.pkl'), 'rb') as file:
            self.best_metrics = pickle.load(file)

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
            batch_sampler = DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)

            steps_per_epoch = len(batch_sampler)
            timer = Timer(steps_per_epoch * epochs)
            timer.start()

            for i in range(epochs):
                self.current_epoch += 1
                avg_loss = 0
                avg_metrics = defaultdict(int)
                self.model.train()

                for batch_idx, batch in enumerate(loader):
                    loss, metrics = self.training_step(batch, batch_idx)
                    self.optimizer_step(self.current_epoch, batch_idx, self.optimizer, loss)
                    self.optimizer_zero_grad(self.current_epoch, batch_idx, self.optimizer)

                    # calculate metrics and loss
                    avg_loss += loss.numpy()[0]
                    for metric, value in metrics.items():
                        avg_metrics[metric] += value.numpy()[0]

                    timer.count()

                    if (batch_idx + 1) % log_interval == 0 and self.local_rank == 0:
                        lr = self.optimizer.current_step_lr()
                        avg_loss /= log_interval
                        if self.use_vdl:
                            self.log_writer.add_scalar(tag='TRAIN/loss', step=timer.current_step, value=avg_loss)

                        print_msg = 'Epoch={}/{}, Step={}/{}'.format(self.current_epoch, epochs, batch_idx + 1,
                                                                     steps_per_epoch)
                        print_msg += ' loss={:.4f}'.format(avg_loss)

                        for metric, value in avg_metrics.items():
                            value /= log_interval
                            if self.use_vdl:
                                self.log_writer.add_scalar(
                                    tag='TRAIN/{}'.format(metric), step=timer.current_step, value=value)
                            print_msg += ' {}={:.4f}'.format(metric, value)

                        print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(lr, timer.timing, timer.eta)

                        logger.train(print_msg)

                        avg_loss = 0
                        avg_metrics = defaultdict(int)

                    if self.current_epoch % save_interval == 0 and batch_idx + 1 == steps_per_epoch and self.local_rank == 0:
                        if eval_dataset:
                            result = self.evaluate(eval_dataset, batch_size, num_workers)
                            eval_loss = result.get('loss', None)
                            eval_metrics = result.get('metrics', {})
                            if self.use_vdl:
                                if eval_loss:
                                    self.log_writer.add_scalar(
                                        tag='EVAL/loss', step=timer.current_step, value=eval_loss)

                                for metric, value in eval_metrics.items():
                                    self.log_writer.add_scalar(
                                        tag='EVAL/{}'.format(metric), step=timer.current_step, value=value)

                            if not self.best_metrics or self.compare_metrics(self.best_metrics, eval_metrics):
                                self.best_metrics = eval_metrics
                                best_model_path = os.path.join(self.checkpoint_dir, 'best_model')
                                self.save_model(best_model_path)
                                self._save_metrics()

                                metric_msg = [
                                    '{}={:.4f}'.format(metric, value) for metric, value in self.best_metrics.items()
                                ]
                                metric_msg = ' '.join(metric_msg)
                                logger.eval('Saving best model to {} [best {}]'.format(best_model_path, metric_msg))

                        self._save_checkpoint()

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
            batch_sampler = DistributedBatchSampler(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

            loader = DataLoader(
                eval_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)

            self.model.eval()
            avg_loss = num_samples = 0
            sum_metrics = defaultdict(int)
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
                    sum_metrics[metric] += value.numpy()[0] * bs

            # print avg metrics and loss
            print_msg = '[Evaluation result]'
            if loss:
                avg_loss /= num_samples
                print_msg += ' avg_loss={:.4f}'.format(avg_loss)

            for metric, value in sum_metrics.items():
                avg_metrics[metric] = value / num_samples
                print_msg += ' avg_{}={:.4f}'.format(metric, avg_metrics[metric])

            logger.eval(print_msg)

            if loss:
                return {'loss': avg_loss, 'metrics': avg_metrics}
            return {'metrics': avg_metrics}

    def training_step(self, batch: Any, batch_idx: int):
        if self.nranks > 1:
            result = self.model._layers.training_step(batch, batch_idx)
        else:
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

    def validation_step(self, batch: Any, batch_idx: int):
        if self.nranks > 1:
            result = self.model._layers.validation_step(batch, batch_idx)
        else:
            result = self.model.validation_step(batch, batch_idx)
        return result

    def optimizer_step(self, current_epoch: int, batch_idx: int, optimizer: fluid.optimizer.Optimizer,
                       loss: fluid.core.VarBase):
        self.optimizer.minimize(loss)

    def optimizer_zero_grad(self, current_epoch: int, batch_idx: int, optimizer: fluid.optimizer.Optimizer):
        self.model.clear_gradients()

    def _compare_metrics(self, old_metric: dict, new_metric: dict):
        '''Compare the whether the new metric value is better than the old one'''
        mainkey = list(new_metric.keys())[0]
        return old_metric[mainkey] < new_metric[mainkey]
