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

import time

import paddle.fluid as fluid
import numpy as np

from paddlehub.common.logger import logger
import paddlehub as hub


def evaluate_cls_task(task, data_reader, feed_list, phase="test", config=None):
    logger.info("Evaluation on {} dataset start".format(phase))
    test_program = task.test_program()
    main_program = task.main_program()
    loss = task.variable("loss")
    accuracy = task.variable("accuracy")
    batch_size = config.batch_size
    place, dev_count = hub.common.get_running_device_info(config)
    exe = fluid.Executor(place=place)
    with fluid.program_guard(test_program):
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        num_eval_examples = acc_sum = loss_sum = 0
        test_reader = data_reader.data_generator(
            batch_size=batch_size, phase=phase)
        eval_time_begin = time.time()
        eval_step = 0
        for batch in test_reader():
            num_batch_examples = len(batch)
            eval_step += 1
            loss_v, accuracy_v = exe.run(
                feed=data_feeder.feed(batch),
                fetch_list=[loss.name, accuracy.name])
            num_eval_examples += num_batch_examples
            if num_eval_examples % 10000 == 0:
                logger.info("{} examples evaluated.".format(num_eval_examples))
            acc_sum += accuracy_v * num_batch_examples
            loss_sum += loss_v * num_batch_examples
        eval_time_used = time.time() - eval_time_begin

        avg_loss = loss_sum / num_eval_examples
        avg_acc = acc_sum / num_eval_examples
        eval_speed = eval_step / eval_time_used

    logger.info(
        "[%s dataset evaluation result] loss=%.5f acc=%.5f [step/sec: %.2f]" %
        (phase, avg_loss, avg_acc, eval_speed))

    return avg_loss, avg_acc, eval_speed


def evaluate_seq_label_task(task,
                            data_reader,
                            feed_list,
                            phase="test",
                            config=None):
    fetch_list = [
        task.variable("labels").name,
        task.variable("infers").name,
        task.variable("seq_len").name,
        task.variable("loss").name
    ]
    logger.info("Evaluation on {} dataset start".format(phase))
    test_program = task.test_program()
    batch_size = config.batch_size
    place, dev_count = hub.common.get_running_device_info(config)
    exe = fluid.Executor(place=place)
    # calculate the num of label from probs variable shape
    num_labels = task.variable("probs").shape[1]
    with fluid.program_guard(test_program):
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        num_eval_examples = acc_sum = loss_sum = 0
        test_reader = data_reader.data_generator(
            batch_size=batch_size, phase=phase)
        eval_time_begin = time.time()
        eval_step = 0
        total_label, total_infer, total_correct = 0.0, 0.0, 0.0
        for batch in test_reader():
            num_batch_examples = len(batch)
            eval_step += 1
            np_labels, np_infers, np_lens, _ = exe.run(
                feed=data_feeder.feed(batch), fetch_list=fetch_list)
            label_num, infer_num, correct_num = chunk_eval(
                np_labels, np_infers, np_lens, num_labels, dev_count)

            total_infer += infer_num
            total_label += label_num
            total_correct += correct_num

        precision, recall, f1 = calculate_f1(total_label, total_infer,
                                             total_correct)
        eval_time_used = time.time() - eval_time_begin
        eval_speed = eval_step / eval_time_used
        logger.info(
            "[%s evaluation] F1-Score=%f, precision=%f, recall=%f [step/sec: %.2f]"
            % (phase, f1, precision, recall, eval_speed))

    return f1, precision, recall


# Sequence label evaluation functions
def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):
    def extract_bio_chunk(seq):
        chunks = []
        cur_chunk = None
        null_index = tag_num - 1
        for index in range(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

            else:
                if cur_chunk is None:
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in range(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in range(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0
            while label_index < len(label_chunks) \
                   and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] \
                    < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] \
                    > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    if infer_chunks[infer_index]["en"] \
                        == label_chunks[label_index]["en"] \
                        and infer_chunks[infer_index]["type"] \
                        == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct


def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
