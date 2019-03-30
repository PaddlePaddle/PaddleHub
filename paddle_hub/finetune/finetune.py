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
from visualdl import LogWriter

from paddle_hub.tools.logger import logger
from paddle_hub.finetune.optimization import bert_finetune
from paddle_hub.finetune.checkpoint import load_checkpoint, save_checkpoint

CKPT_FILE = "ckpt.meta"


def _finetune_model(task,
                    data_processor,
                    feed_list,
                    config=None,
                    eval_model=False):
    main_program = task.main_program()
    startup_program = task.startup_program()
    loss = task.variable("loss")
    accuracy = task.variable("accuracy")

    epoch = config.num_epoch
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    use_cuda = config.use_cuda
    with_memory_optimization = config.with_memory_optimization
    checkpoint_dir = config.checkpoint_dir
    checkpoint_path = os.path.join(checkpoint_dir, CKPT_FILE)
    log_writter = LogWriter(
        os.path.join(checkpoint_dir, "vdllog"), sync_cycle=10)

    with fluid.program_guard(main_program, startup_program):
        if use_cuda:
            place = fluid.CUDAPlace(0)
            dev_count = fluid.core.get_cuda_device_count()
        else:
            place = fluid.CPUPlace()
            dev_count = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        exe = fluid.Executor(place=place)
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)

        if config.finetune_strategy == "bert_finetune":
            scheduled_lr = bert_finetune(task, main_program, data_processor,
                                         config, dev_count)
        elif config.optimizer == "adam":
            optimizer = fluid.optimizer.Adam(learning_rate=config.learning_rate)
            optimizer.minimize(loss)
        #TODO: add more finetune strategy

        if with_memory_optimization:
            logger.info("Memory optimization start...")
            optimize_time_begin = time.time()
            fluid.memory_optimize(
                input_program=fluid.default_main_program(),
                skip_opt_set=[
                    # skip task graph variable memory optimization
                    loss.name,
                    accuracy.name
                ])
            time_used = time.time() - optimize_time_begin
            logger.info(
                "Memory optimization done! Time elapsed %f sec" % time_used)

        lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
            program=main_program, batch_size=batch_size)
        logger.info("Theoretical memory usage in training: %.3f - %.3f %s" %
                    (lower_mem, upper_mem, unit)),
        # initilize
        if os.path.exists(checkpoint_path):
            last_epoch, step, last_model_dir = load_checkpoint(checkpoint_path)
            fluid.io.load_persistables(exe, last_model_dir)
        else:
            exe.run(fluid.default_startup_program())
            step = 0
            last_epoch = 0
        best_eval_acc = 0
        logger.info("Finetune start")

        # add visualdl scalar
        with log_writter.mode("train") as logw:
            train_loss_scalar = logw.scalar(tag="loss[train]")
            train_acc_scalar = logw.scalar(tag="accuracy[train]")
        with log_writter.mode("evaluate") as logw:
            eval_loss_scalar = logw.scalar(tag="loss[evaluate]")
            eval_acc_scalar = logw.scalar(tag="accuracy[evaluate]")

        train_time_begin = time.time()
        for index in range(last_epoch, epoch):
            train_reader = data_processor.data_generator(
                batch_size=batch_size, phase='train')
            size = accuracy_sum = loss_sum = 0
            for batch in train_reader():
                loss_v, accuracy_v = exe.run(
                    feed=data_feeder.feed(batch),
                    fetch_list=[loss.name, accuracy.name])
                step += 1
                size += len(batch)
                accuracy_sum += accuracy_v * len(batch)
                loss_sum += loss_v * len(batch)

                # print log
                if step % config.log_interval == 0:
                    train_time_used = time.time() - train_time_begin
                    speed = config.log_interval / train_time_used
                    train_time_begin = time.time()
                    logger.info(
                        "step %d: loss=%.5f acc=%.5f [step/sec: %.2f]" %
                        (step, loss_sum / size, accuracy_sum / size, speed))

                    # record visualdl log
                    record_step = step
                    train_loss_scalar.add_record(record_step, loss_sum / size)
                    train_acc_scalar.add_record(record_step,
                                                accuracy_sum / size)

                    size = accuracy_sum = loss_sum = 0

                if step % config.save_ckpt_interval == 0:
                    model_save_dir = os.path.join(checkpoint_dir,
                                                  "model_in_step_%d" % step)
                    fluid.io.save_persistables(exe, dirname=model_save_dir)
                    save_checkpoint(
                        checkpoint_path,
                        last_epoch=index,
                        last_step=step,
                        last_model_dir=model_save_dir)

                if eval_model and step % config.eval_interval == 0:
                    eval_loss, eval_acc, eval_perf = evaluate(
                        task,
                        data_processor,
                        feed_list,
                        phase="validate",
                        config=config)
                    record_step = step
                    eval_loss_scalar.add_record(record_step, eval_loss)
                    eval_acc_scalar.add_record(record_step, eval_acc)
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        model_save_dir = os.path.join(checkpoint_dir,
                                                      "model_best")
                        fluid.io.save_persistables(exe, dirname=model_save_dir)

        # update model and checkpoint
        model_save_dir = os.path.join(checkpoint_dir, "model_latest")
        fluid.io.save_persistables(exe, dirname=model_save_dir)
        save_checkpoint(
            checkpoint_path,
            last_epoch=epoch + 1,
            last_step=step,
            last_model_dir=model_save_dir)
        # eval before end
        if eval_model:
            evaluate(
                task, data_processor, feed_list, phase="test", config=config)
        logger.info("Finetune finished")


def finetune_and_eval(task, data_processor, feed_list, config=None):
    _finetune_model(task, data_processor, feed_list, config, eval_model=True)


def finetune(task, data_processor, feed_list, config=None):
    _finetune_model(task, data_processor, feed_list, config, eval_model=False)


def evaluate(task, data_processor, feed_list, phase="test", config=None):
    inference_program = task.inference_program()
    main_program = task.main_program()
    loss = task.variable("loss")
    accuracy = task.variable("accuracy")
    use_cuda = config.use_cuda
    batch_size = config.batch_size
    with fluid.program_guard(inference_program):
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        exe = fluid.Executor(place=place)
        size = accuracy_sum = loss_sum = 0
        test_reader = data_processor.data_generator(
            batch_size=batch_size, phase=phase)
        eval_time_begin = time.time()
        for index, batch in enumerate(test_reader()):
            loss_v, accuracy_v, = exe.run(
                feed=data_feeder.feed(batch), fetch_list=[loss, accuracy.name])
            size += len(batch)
            accuracy_sum += accuracy_v * len(batch)
            loss_sum += loss_v * len(batch)
        eval_time_used = time.time() - eval_time_begin
        eval_speed = index / eval_time_used
    logger.info("[Evaluation] loss=%.5f acc=%.5f [step/sec: %.2f]" %
                (loss_sum / size, accuracy_sum / size, eval_speed))

    return loss_sum / size, accuracy_sum / size, eval_speed
