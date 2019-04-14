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
"""Finetuning on classification task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import paddle
import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    input_dict, output_dict, program = module.context(
        max_seq_len=args.max_seq_len)

    # Sentence classification  dataset reader
    dataset = hub.dataset.ChnSentiCorp()
    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    with fluid.program_guard(program):
        label = fluid.layers.data(name="label", shape=[1], dtype='int64')

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        pooled_output = output_dict["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of ERNIE's module need
        feed_list = [
            input_dict["input_ids"].name, input_dict["position_ids"].name,
            input_dict["segment_ids"].name, input_dict["input_mask"].name,
            label.name
        ]

        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.create_text_classification_task(
            feature=pooled_output, label=label, num_classes=dataset.num_labels)

        # classificatin probability tensor
        probs = cls_task.variable("probs")

        pred = fluid.layers.argmax(probs, axis=1)

        # load best model checkpoint
        fluid.io.load_persistables(exe, args.checkpoint_dir)

        inference_program = program.clone(for_test=True)

        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        test_reader = reader.data_generator(phase='test', shuffle=False)
        test_examples = dataset.get_test_examples()
        total = 0
        correct = 0
        for index, batch in enumerate(test_reader()):
            pred_v = exe.run(
                feed=data_feeder.feed(batch),
                fetch_list=[pred.name],
                program=inference_program)
            total += 1
            if (pred_v[0][0] == int(test_examples[index].label)):
                correct += 1
                acc = 1.0 * correct / total
            print("%s\tpredict=%s" % (test_examples[index], pred_v[0][0]))
        print("accuracy = %f" % acc)
