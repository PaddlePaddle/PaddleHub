#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Finetuning on sequence labeling task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import numpy as np
import os
import time

import paddle
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.finetune.evaluate import chunk_eval, calculate_f1

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    input_dict, output_dict, program = module.context(
        max_seq_len=args.max_seq_len)

    # Sentence labeling dataset reader
    dataset = hub.dataset.MSRA_NER()
    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)
    inv_label_map = {val: key for key, val in reader.label_map.items()}

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    with fluid.program_guard(program):
        # Use "sequence_outputs" for token-level output.
        sequence_output = output_dict["sequence_output"]

        # Define a classfication finetune task by PaddleHub's API
        seq_label_task = hub.create_seq_label_task(
            feature=sequence_output,
            num_classes=dataset.num_labels,
            max_seq_len=args.max_seq_len)

        # Setup feed list for data feeder
        # Must feed all the tensor of ERNIE's module need
        # Compared to classification task, we need add seq_len tensor to feedlist
        feed_list = [
            input_dict["input_ids"].name, input_dict["position_ids"].name,
            input_dict["segment_ids"].name, input_dict["input_mask"].name,
            seq_label_task.variable('label').name,
            seq_label_task.variable('seq_len').name
        ]

        fetch_list = [
            seq_label_task.variable("labels").name,
            seq_label_task.variable("infers").name,
            seq_label_task.variable("seq_len").name
        ]

        # classification probability tensor
        probs = seq_label_task.variable("probs")

        # load best model checkpoint
        fluid.io.load_persistables(exe, args.checkpoint_dir)

        inference_program = program.clone(for_test=True)

        # calculate the num of label from probs variable shape
        num_labels = seq_label_task.variable("probs").shape[1]

        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        test_reader = reader.data_generator(phase='test', shuffle=False)
        test_examples = dataset.get_test_examples()
        total_label, total_infer, total_correct = 0.0, 0.0, 0.0
        for index, batch in enumerate(test_reader()):
            np_labels, np_infers, np_lens = exe.run(
                feed=data_feeder.feed(batch),
                fetch_list=fetch_list,
                program=inference_program)
            label_num, infer_num, correct_num = chunk_eval(
                np_labels, np_infers, np_lens, num_labels)

            total_infer += infer_num
            total_label += label_num
            total_correct += correct_num

            labels = np_labels.reshape([-1]).astype(np.int32).tolist()
            label_str = ""
            count = 0
            for label_val in labels:
                label_str += inv_label_map[label_val]
                count += 1
                if count == np_lens:
                    break

            print("%s\tpredict=%s" % (test_examples[index], label_str))

        precision, recall, f1 = calculate_f1(total_label, total_infer,
                                             total_correct)
        print("F1-Score=%f, precision=%f, recall=%f " % (f1, precision, recall))
