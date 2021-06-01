# -*- coding: utf-8 -*-
#*******************************************************************************
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
#*******************************************************************************
"""

Authors: lvhaijun01@baidu.com
Date:     2020-11-24 20:43
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six
import os
from typing import Dict, List, Optional, Union, Tuple
from auto_augment.autoaug.utils import log
import logging
logger = log.get_logger(level=logging.INFO)
import auto_augment
auto_augment_path = auto_augment.__file__


class HubFitterClassifer(object):
    """Trains an instance of the Model class."""

    def __init__(self, hparams: dict) -> None:
        """
        定义分类任务的数据、模型

        Args:
            hparams:
        """

        def set_paddle_flags(**kwargs):
            for key, value in kwargs.items():
                if os.environ.get(key, None) is None:
                    os.environ[key] = str(value)

        # NOTE(paddle-dev): All of these flags should be set before
        # `import paddle`. Otherwise, it would not take any effect.
        set_paddle_flags(
            # enable GC to save memory
            FLAGS_fraction_of_gpu_memory_to_use=hparams.resource_config.gpu, )
        import paddle
        import paddlehub as hub
        from paddlehub_utils.trainer import CustomTrainer
        from paddlehub_utils.reader import _init_loader

        # todo now does not support fleet distribute training
        # from paddle.fluid.incubate.fleet.base import role_maker
        # from paddle.fluid.incubate.fleet.collective import fleet
        # role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        # fleet.init(role)

        logger.info("classficiation data augment search begin")
        self.hparams = hparams
        # param compatible
        self._fit_param(show=True)
        paddle.disable_static(paddle.CUDAPlace(paddle.distributed.get_rank()))

        train_dataset, eval_dataset = _init_loader(self.hparams)
        model = hub.Module(
            name=hparams["task_config"]["classifier"]["model_name"],
            label_list=self.class_to_id_dict.keys(),
            load_checkpoint=None)

        optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        trainer = CustomTrainer(model=model, optimizer=optimizer, checkpoint_dir='img_classification_ckpt')
        self.model = model
        self.optimizer = optimizer

        trainer.init_train_and_eval(
            train_dataset, epochs=100, batch_size=32, eval_dataset=eval_dataset, save_interval=1)
        self.trainer = trainer

    def _fit_param(self, show: bool = False) -> None:
        """
        param fit
        Args:
            hparams:

        Returns:

        """
        hparams = self.hparams
        self._get_label_info(hparams)

    def _get_label_info(self, hparams: dict) -> None:
        """

        Args:
            hparams:

        Returns:

        """
        from paddlehub_utils.reader import _read_classes
        data_config = hparams.data_config
        label_list = data_config.label_list
        if os.path.isfile(label_list):
            class_to_id_dict = _read_classes(label_list)
        else:
            assert 0, "label_list:{} not exist".format(label_list)
        self.num_classes = len(class_to_id_dict)
        self.class_to_id_dict = class_to_id_dict

    def reset_config(self, new_hparams: dict) -> None:
        """
        reset config, used by search stage
        Args:
            new_hparams:

        Returns:

        """
        self.hparams = new_hparams
        self.trainer.train_loader.dataset.reset_policy(new_hparams.search_space)
        return None

    def save_model(self, checkpoint_dir: str, step: Optional[str] = None) -> str:
        """Dumps model into the backup_dir.

        Args:
          step: If provided, creates a checkpoint with the given step
            number, instead of overwriting the existing checkpoints.
        """
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch') + '-' + str(step)
        logger.info('Saving model checkpoint to {}'.format(checkpoint_path))
        self.trainer.save_model(os.path.join(checkpoint_path, "checkpoint"))

        return checkpoint_path

    def extract_model_spec(self, checkpoint_path: str) -> None:
        """Loads a checkpoint with the architecture structure stored in the name."""
        ckpt_path = os.path.join(checkpoint_path, "checkpoint")
        self.trainer.load_model(ckpt_path)
        logger.info('Loaded child model checkpoint from {}'.format(checkpoint_path))

    def eval_child_model(self, mode: str, pass_id: int = 0) -> dict:
        """Evaluate the child model.

        Args:
          model: image model that will be evaluated.
          data_loader: dataset object to extract eval data from.
          mode: will the model be evalled on train, val or test.

        Returns:
          Accuracy of the model on the specified dataset.
        """
        eval_loader = self.trainer.eval_loader
        res = self.trainer.evaluate_process(eval_loader)
        top1_acc = res["metrics"]["acc"]

        if mode == "val":
            return {"val_acc": top1_acc}
        elif mode == "test":
            return {"test_acc": top1_acc}
        else:
            raise NotImplementedError

    def train_one_epoch(self, pass_id: int) -> dict:
        """

        Args:
            model:
            train_loader:
            optimizer:

        Returns:

        """
        from paddlehub.utils.utils import Timer

        batch_sampler = self.trainer.batch_sampler
        train_loader = self.trainer.train_loader
        steps_per_epoch = len(batch_sampler)
        task_config = self.hparams.task_config
        task_type = task_config.task_type
        epochs = task_config.classifier.epochs
        timer = Timer(steps_per_epoch * epochs)
        timer.start()
        self.trainer.train_one_epoch(
            loader=train_loader,
            timer=timer,
            current_epoch=pass_id,
            epochs=epochs,
            log_interval=10,
            steps_per_epoch=steps_per_epoch)
        return {"train_acc": 0}

    def _run_training_loop(self, curr_epoch: int) -> dict:
        """Trains the model `m` for one epoch."""
        start_time = time.time()
        train_acc = self.train_one_epoch(curr_epoch)
        logger.info('Epoch:{} time(min): {}'.format(curr_epoch, (time.time() - start_time) / 60.0))
        return train_acc

    def _compute_final_accuracies(self, iteration: int) -> dict:
        """Run once training is finished to compute final test accuracy."""
        task_config = self.hparams.task_config
        task_type = task_config.task_type

        if (iteration >= task_config[task_type].epochs - 1):
            test_acc = self.eval_child_model('test', iteration)
            pass
        else:
            test_acc = {"test_acc": 0}
        logger.info('Test acc: {}'.format(test_acc))
        return test_acc

    def run_model(self, epoch: int) -> dict:
        """Trains and evalutes the image model."""
        self._fit_param()
        train_acc = self._run_training_loop(epoch)
        valid_acc = self.eval_child_model(mode="val", pass_id=epoch)
        logger.info('valid acc: {}'.format(valid_acc))
        all_metric = {}
        all_metric.update(train_acc)
        all_metric.update(valid_acc)
        return all_metric
