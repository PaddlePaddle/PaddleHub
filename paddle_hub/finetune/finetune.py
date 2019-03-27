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
import paddle.fluid as fluid
import time
import numpy as np
import multiprocessing

from paddle_hub.finetune.optimization import bert_optimization
from paddle_hub.finetune.config import FinetuneConfig


def finetune_and_eval(task, feed_list, data_processor, config=None):
    # environment setup
    if config.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    # hub.finetune_and_eval start here
    #TODO: to simplify
    loss = task.variable("loss")
    probs = task.variable("probs")
    accuracy = task.variable("accuracy")
    num_example = task.variable("num_example")

    num_train_examples = data_processor.get_num_examples(phase='train')
    if config.in_tokens:
        max_train_steps = config.num_epoch * num_train_examples // (
            config.batch_size // config.max_seq_len) // dev_count
    else:
        max_train_steps = config.num_epoch * num_train_examples // config.batch_size // dev_count

    warmup_steps = int(max_train_steps * config.warmup_proportion)

    # obtain main program from Task class
    train_program = task.main_program()
    startup_program = task.startup_program()
    # clone test program before optimize
    test_program = train_program.clone(for_test=True)

    bert_optimization(loss, warmup_steps, max_train_steps, config.learning_rate,
                      train_program, config.weight_decay)

    # memory optimization
    fluid.memory_optimize(
        input_program=train_program,
        skip_opt_set=[
            # skip task graph variable memory optimization
            loss.name,
            probs.name,
            accuracy.name,
            num_example.name
        ])

    exe.run(startup_program)
    feeder = fluid.DataFeeder(feed_list=feed_list, place=place)

    # Traning block
    # prepare training dataset
    total_loss, total_acc, total_num_example = [], [], []
    step = 0
    time_begin = time.time()
    train_time_used = 0.0
    for epoch in range(1, config.num_epoch + 1):
        print("Epoch {}".format(epoch))
        train_data_generator = data_processor.data_generator(
            batch_size=config.batch_size, phase='train', shuffle=False)
        for example in train_data_generator():
            step += 1
            train_time_begin = time.time()
            np_loss, np_acc, np_num_example = exe.run(
                program=train_program,
                feed=feeder.feed([example]),
                fetch_list=[loss, accuracy, num_example])
            train_time_used += time.time() - train_time_begin

            # Statistic Block
            total_loss.extend(np_loss * np_num_example)
            total_acc.extend(np_acc * np_num_example)
            total_num_example.extend(np_num_example)
            if step % config.log_interval == 0:
                # get training progress
                accum_num_example = np.sum(total_num_example)
                print(
                    "step {}: loss={:.5f} acc={:.5f} [step/sec: {:.2f}]".format(
                        step,
                        np.sum(total_loss) / accum_num_example,
                        np.sum(total_acc) / accum_num_example,
                        config.log_interval / train_time_used))
                # reset statistic variables
                total_loss, total_acc, total_num_example = [], [], []
                train_time_used = 0.0

            # Evaluation block
            if step % config.eval_interval == 0:
                test_data_generator = data_processor.data_generator(
                    batch_size=config.batch_size, phase='test', shuffle=False)
                dev_data_generator = data_processor.data_generator(
                    batch_size=config.batch_size, phase='dev', shuffle=False)
                evaluate(task, test_program, exe, feeder, dev_data_generator)
                evaluate(task, test_program, exe, feeder, test_data_generator)

            # Save model checkpoint
            if step % config.save_ckpt_interval == 0:
                save_checkpoint(exe, train_program, step, config.checkpoint_dir)

    # finish final evaluation on testset
    test_data_generator = data_processor.data_generator(
        batch_size=config.batch_size, phase='test', shuffle=False)
    evaluate(task, test_program, exe, feeder, test_data_generator)


def save_checkpoint(exe, train_program, step, ckpt_dir):
    #TODO: add global step variable for restore checkpoint like tensorflow
    ckpt_step_dir = os.path.join(ckpt_dir, "step_{}".format(step))
    fluid.io.save_persistables(exe, ckpt_step_dir, train_program)


def evaluate(task, test_program, exe, feeder, data_generator):
    loss = task.variable("loss")
    probs = task.variable("probs")
    accuracy = task.variable("accuracy")
    num_example = task.variable("num_example")

    total_loss, total_acc, total_num_example = [], [], []
    eval_step = 0
    eval_time_begin = time.time()
    for example in data_generator():
        eval_step += 1
        np_loss, np_acc, np_num_example = exe.run(
            program=test_program,
            feed=feeder.feed([example]),
            fetch_list=[loss, accuracy, num_example])
        total_loss.extend(np_loss * np_num_example)
        total_acc.extend(np_acc * np_num_example)
        total_num_example.extend(np_num_example)
    eval_time_used = time.time() - eval_time_begin
    accum_num_example = np.sum(total_num_example)
    print("[evaluation] loss={:.5f} acc={:.5f} [step/sec: {:.2f}]".format(
        np.sum(total_loss) / accum_num_example,
        np.sum(total_acc) / accum_num_example, eval_step / eval_time_used))
