# -*- coding: utf-8 -*-
#*******************************************************************************
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
#*******************************************************************************
"""

Authors: lvhaijun01@baidu.com
Date:     2020-11-24 20:46
"""

from paddlehub.finetune.trainer import Trainer
import os
from collections import defaultdict
import paddle
from paddle.distributed import ParallelEnv
from paddlehub.utils.log import logger
from paddlehub.utils.utils import Timer


class CustomTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super(CustomTrainer, self).__init__(**kwargs)

    def init_train_and_eval(self,
                            train_dataset: paddle.io.Dataset,
                            epochs: int = 1,
                            batch_size: int = 1,
                            num_workers: int = 0,
                            eval_dataset: paddle.io.Dataset = None,
                            log_interval: int = 10,
                            save_interval: int = 10) -> None:
        self.batch_sampler, self.train_loader = self.init_train(train_dataset, batch_size, num_workers)
        self.eval_loader = self.init_evaluate(eval_dataset, batch_size, num_workers)

    def init_train(self, train_dataset: paddle.io.Dataset, batch_size: int = 1, num_workers: int = 0) -> tuple:
        use_gpu = True
        place = paddle.CUDAPlace(ParallelEnv().dev_id) if use_gpu else paddle.CPUPlace()
        paddle.disable_static(place)

        batch_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        loader = paddle.io.DataLoader(
            train_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)
        return batch_sampler, loader

    def train_one_epoch(self, loader: paddle.io.DataLoader, timer: Timer, current_epoch: int, epochs: int,
                        log_interval: int, steps_per_epoch: int) -> None:
        avg_loss = 0
        avg_metrics = defaultdict(int)
        self.model.train()

        for batch_idx, batch in enumerate(loader):
            loss, metrics = self.training_step(batch, batch_idx)
            self.optimizer_step(current_epoch, batch_idx, self.optimizer, loss)
            self.optimizer_zero_grad(current_epoch, batch_idx, self.optimizer)

            # calculate metrics and loss
            avg_loss += loss.numpy()[0]
            for metric, value in metrics.items():
                avg_metrics[metric] += value.numpy()[0]

            timer.count()

            if (batch_idx + 1) % log_interval == 0 and self.local_rank == 0:
                lr = self.optimizer.get_lr()
                avg_loss /= log_interval
                if self.use_vdl:
                    self.log_writer.add_scalar(tag='TRAIN/loss', step=timer.current_step, value=avg_loss)

                print_msg = 'Epoch={}/{}, Step={}/{}'.format(current_epoch, epochs, batch_idx + 1, steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)

                for metric, value in avg_metrics.items():
                    value /= log_interval
                    if self.use_vdl:
                        self.log_writer.add_scalar(tag='TRAIN/{}'.format(metric), step=timer.current_step, value=value)
                    print_msg += ' {}={:.4f}'.format(metric, value)

                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(lr, timer.timing, timer.eta)

                logger.train(print_msg)

                avg_loss = 0
                avg_metrics = defaultdict(int)

    def train(self,
              train_dataset: paddle.io.Dataset,
              epochs: int = 1,
              batch_size: int = 1,
              num_workers: int = 0,
              eval_dataset: paddle.io.Dataset = None,
              log_interval: int = 10,
              save_interval: int = 10):
        '''
        Train a model with specific config.

        Args:
            train_dataset(paddle.io.Dataset) : Dataset to train the model
            epochs(int) : Number of training loops, default is 1.
            batch_size(int) : Batch size of per step, default is 1.
            num_workers(int) : Number of subprocess to load data, default is 0.
            eval_dataset(paddle.io.Dataset) : The validation dataset, deafult is None. If set, the Trainer will
                execute evaluate function every `save_interval` epochs.
            log_interval(int) : Log the train infomation every `log_interval` steps.
            save_interval(int) : Save the checkpoint every `save_interval` epochs.
        '''
        batch_sampler, loader = self.init_train(train_dataset, batch_size, num_workers)
        steps_per_epoch = len(batch_sampler)
        timer = Timer(steps_per_epoch * epochs)
        timer.start()

        for i in range(epochs):
            loader.dataset.set_epoch(epochs)
            self.current_epoch += 1
            self.train_one_epoch(loader, timer, self.current_epoch, epochs, log_interval, steps_per_epoch)

            # todo, why paddlehub put save, eval in batch?
            if self.current_epoch % save_interval == 0 and self.local_rank == 0:
                if eval_dataset:
                    result = self.evaluate(eval_dataset, batch_size, num_workers)
                    eval_loss = result.get('loss', None)
                    eval_metrics = result.get('metrics', {})
                    if self.use_vdl:
                        if eval_loss:
                            self.log_writer.add_scalar(tag='EVAL/loss', step=timer.current_step, value=eval_loss)

                        for metric, value in eval_metrics.items():
                            self.log_writer.add_scalar(
                                tag='EVAL/{}'.format(metric), step=timer.current_step, value=value)

                    if not self.best_metrics or self.compare_metrics(self.best_metrics, eval_metrics):
                        self.best_metrics = eval_metrics
                        best_model_path = os.path.join(self.checkpoint_dir, 'best_model')
                        self.save_model(best_model_path)
                        self._save_metrics()

                        metric_msg = ['{}={:.4f}'.format(metric, value) for metric, value in self.best_metrics.items()]
                        metric_msg = ' '.join(metric_msg)
                        logger.eval('Saving best model to {} [best {}]'.format(best_model_path, metric_msg))

                self._save_checkpoint()

    def init_evaluate(self, eval_dataset: paddle.io.Dataset, batch_size: int, num_workers: int) -> paddle.io.DataLoader:
        use_gpu = True
        place = paddle.CUDAPlace(ParallelEnv().dev_id) if use_gpu else paddle.CPUPlace()
        paddle.disable_static(place)

        batch_sampler = paddle.io.DistributedBatchSampler(
            eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        loader = paddle.io.DataLoader(
            eval_dataset, batch_sampler=batch_sampler, places=place, num_workers=num_workers, return_list=True)
        return loader

    def evaluate_process(self, loader: paddle.io.DataLoader) -> dict:
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

    def evaluate(self, eval_dataset: paddle.io.Dataset, batch_size: int = 1, num_workers: int = 0) -> dict:
        '''
        Run evaluation and returns metrics.

        Args:
            eval_dataset(paddle.io.Dataset) : The validation dataset
            batch_size(int) : Batch size of per step, default is 1.
            num_workers(int) : Number of subprocess to load data, default is 0.
        '''

        loader = self.init_evaluate(eval_dataset, batch_size, num_workers)
        res = self.evaluate_process(loader)
        return res
