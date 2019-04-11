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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import paddle
import paddle.fluid as fluid
import numpy as np
from visualdl import LogWriter

from paddlehub.common.logger import logger
from paddlehub.finetune.strategy import BERTFinetuneStrategy, DefaultStrategy
from paddlehub.finetune.checkpoint import load_checkpoint, save_checkpoint
from paddlehub.finetune.evaluate import evaluate_cls_task, evaluate_seq_labeling_task
import paddlehub as hub


def _do_memory_optimization(task, config):
    if config.enable_memory_optim:
        logger.info("Memory optimization start...")
        task_var_name = task.metric_variable_names()
        logger.info(
            "Skip memory optimization on variables: {}".format(task_var_name))
        optimize_time_begin = time.time()
        fluid.memory_optimize(
            input_program=fluid.default_main_program(),
            # skip memory optimization on task metric variables
            skip_opt_set=task_var_name)
        time_used = time.time() - optimize_time_begin
        logger.info("Memory optimization done! Time elapsed %f sec" % time_used)

    lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
        program=fluid.default_main_program(), batch_size=config.batch_size)
    logger.info("Theoretical memory usage in training: %.2f - %.2f %s" %
                (lower_mem, upper_mem, unit)),


def _finetune_seq_label_task(task,
                             data_reader,
                             feed_list,
                             config=None,
                             do_eval=False):
    """
    Finetune sequence labeling task, evaluate metric is F1, precision and recall

    """
    main_program = task.main_program()
    startup_program = task.startup_program()
    loss = task.variable("loss")
    seq_len = task.variable("seq_len")

    num_epoch = config.num_epoch
    batch_size = config.batch_size

    place, dev_count = hub.common.get_running_device_info(config)
    with fluid.program_guard(main_program, startup_program):
        exe = fluid.Executor(place=place)
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)

        # Select strategy
        if isinstance(config.strategy, hub.BERTFinetuneStrategy):
            scheduled_lr = config.strategy.execute(loss, main_program,
                                                   data_reader, config)
        elif isinstance(config.strategy, hub.DefaultStrategy):
            config.strategy.execute(loss)
        #TODO: add more finetune strategy

        _do_memory_optimization(task, config)

        # Try to restore model training checkpoint
        current_epoch, global_step = load_checkpoint(config.checkpoint_dir, exe)

        train_time_used = 0
        logger.info("PaddleHub finetune start")

        exe.run(fluid.default_startup_program())

        # Finetune loop
        for epoch in range(current_epoch, num_epoch + 1):
            train_reader = data_reader.data_generator(
                batch_size=batch_size, phase='train')
            num_trained_examples = loss_sum = 0
            for batch in train_reader():
                num_batch_examples = len(batch)
                train_time_begin = time.time()
                loss_v = exe.run(
                    feed=data_feeder.feed(batch), fetch_list=[loss.name])
                train_time_used += time.time() - train_time_begin
                global_step += 1
                num_trained_examples += num_batch_examples
                loss_sum += loss_v[0] * num_batch_examples

                # log fintune status
                if global_step % config.log_interval == 0:
                    avg_loss = loss_sum / num_trained_examples
                    speed = config.log_interval / train_time_used
                    logger.info("step %d: loss=%.5f [step/sec: %.2f]" %
                                (global_step, avg_loss, speed))

                    train_time_used = 0
                    num_trained_examples = loss_sum = 0

                if config.save_ckpt_interval and global_step % config.save_ckpt_interval == 0:
                    # NOTE: current saved checkpoint machanism is not completed,
                    # it can't restore correct dataset training status
                    save_checkpoint(
                        checkpoint_dir=config.checkpoint_dir,
                        current_epoch=epoch,
                        global_step=global_step,
                        exe=exe)

                if do_eval and global_step % config.eval_interval == 0:
                    evaluate_seq_labeling_task(
                        task,
                        data_reader,
                        feed_list,
                        phase="test",
                        config=config)
                    evaluate_seq_labeling_task(
                        task,
                        data_reader,
                        feed_list,
                        phase="dev",
                        config=config)

        # NOTE: current saved checkpoint machanism is not completed, it can't
        # resotre dataset training status
        save_checkpoint(
            checkpoint_dir=config.checkpoint_dir,
            current_epoch=num_epoch + 1,
            global_step=global_step,
            exe=exe)

        if do_eval:
            evaluate_seq_labeling_task(
                task, data_reader, feed_list, phase="dev", config=config)
            evaluate_seq_labeling_task(
                task, data_reader, feed_list, phase="test", config=config)
        logger.info("PaddleHub finetune finished.")


def _finetune_cls_task(task, data_reader, feed_list, config=None,
                       do_eval=False):
    main_program = task.main_program()
    startup_program = task.startup_program()
    loss = task.variable("loss")
    accuracy = task.variable("accuracy")

    num_epoch = config.num_epoch
    batch_size = config.batch_size
    log_writter = LogWriter(
        os.path.join(config.checkpoint_dir, "vdllog"), sync_cycle=10)

    place, dev_count = hub.common.get_running_device_info(config)
    with fluid.program_guard(main_program, startup_program):
        exe = fluid.Executor(place=place)
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)

        # select strategy
        if isinstance(config.strategy, hub.BERTFinetuneStrategy):
            scheduled_lr = config.strategy.execute(loss, main_program,
                                                   data_reader, config)
        elif isinstance(config.strategy, hub.DefaultStrategy):
            config.strategy.execute(loss)
        #TODO: add more finetune strategy

        _do_memory_optimization(task, config)

        # Try to restore model training checkpoint
        current_epoch, global_step = load_checkpoint(config.checkpoint_dir, exe)

        best_eval_acc = 0.0
        train_time_used = 0
        logger.info("PaddleHub finetune start")

        # add visualdl scalar
        with log_writter.mode("train") as logw:
            train_loss_scalar = logw.scalar(tag="loss[train]")
            train_acc_scalar = logw.scalar(tag="accuracy[train]")
        with log_writter.mode("evaluate") as logw:
            eval_loss_scalar = logw.scalar(tag="loss[evaluate]")
            eval_acc_scalar = logw.scalar(tag="accuracy[evaluate]")

        exe.run(fluid.default_startup_program())

        # Finetune loop
        for epoch in range(current_epoch, num_epoch + 1):
            train_reader = data_reader.data_generator(
                batch_size=batch_size, phase='train')
            num_trained_examples = acc_sum = loss_sum = 0
            for batch in train_reader():
                num_batch_examples = len(batch)
                train_time_begin = time.time()
                loss_v, accuracy_v = exe.run(
                    feed=data_feeder.feed(batch),
                    fetch_list=[loss.name, accuracy.name])
                train_time_used += time.time() - train_time_begin
                global_step += 1
                num_trained_examples += num_batch_examples
                acc_sum += accuracy_v * num_batch_examples
                loss_sum += loss_v * num_batch_examples

                # log fintune status
                if global_step % config.log_interval == 0:
                    avg_loss = loss_sum / num_trained_examples
                    avg_acc = acc_sum / num_trained_examples
                    speed = config.log_interval / train_time_used
                    logger.info("step %d: loss=%.5f acc=%.5f [step/sec: %.2f]" %
                                (global_step, avg_loss, avg_acc, speed))

                    # record visualdl log
                    train_loss_scalar.add_record(global_step, avg_loss)
                    train_acc_scalar.add_record(global_step, avg_acc)

                    train_time_used = 0
                    num_trained_examples = acc_sum = loss_sum = 0

                if config.save_ckpt_interval and global_step % config.save_ckpt_interval == 0:
                    # NOTE: current saved checkpoint machanism is not completed,
                    # it can't restore dataset training status
                    save_checkpoint(
                        checkpoint_dir=config.checkpoint_dir,
                        current_epoch=epoch,
                        global_step=global_step,
                        exe=exe)

                if do_eval and global_step % config.eval_interval == 0:
                    eval_loss, eval_acc, eval_perf = evaluate_cls_task(
                        task,
                        data_reader,
                        feed_list,
                        phase="val",
                        config=config)
                    eval_loss_scalar.add_record(global_step, eval_loss)
                    eval_acc_scalar.add_record(global_step, eval_acc)
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        model_saved_dir = os.path.join(config.checkpoint_dir,
                                                       "best_model")
                        logger.info(
                            "best model saved to %s [best accuracy=%.5f]" %
                            (model_saved_dir, best_eval_acc))
                        fluid.io.save_persistables(exe, dirname=model_saved_dir)

        # NOTE: current saved checkpoint machanism is not completed, it can't
        # resotre dataset training status
        save_checkpoint(
            checkpoint_dir=config.checkpoint_dir,
            current_epoch=num_epoch + 1,
            global_step=global_step,
            exe=exe)

        if do_eval:
            evaluate_cls_task(
                task, data_reader, feed_list, phase="test", config=config)
        logger.info("PaddleHub finetune finished.")


def finetune_and_eval(task, data_reader, feed_list, config=None):
    if task.task_type == "sequence_labeling":
        _finetune_seq_label_task(
            task, data_reader, feed_list, config, do_eval=True)
    # if it's image_classification and text classificaiton
    else:
        _finetune_cls_task(task, data_reader, feed_list, config, do_eval=True)


def finetune(task, data_reader, feed_list, config=None):
    _finetune_cls_task(task, data_reader, feed_list, config, do_eval=False)
