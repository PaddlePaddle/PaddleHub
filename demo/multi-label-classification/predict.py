#coding:utf-8
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

import argparse
import ast
import numpy as np
import os
import time

import paddle
import paddle.fluid as fluid
import paddlehub as hub

import pandas as pd

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub BERT pretrained model
    module = hub.Module(name="ernie_eng_base.hub_module")

    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Setup feed list for data feeder
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Download dataset and use MultiLabelReader to read dataset
    dataset = hub.dataset.Toxic()

    reader = hub.reader.MultiLabelClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a classfication finetune task by PaddleHub's API
    multi_label_cls_task = hub.MultiLabelClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config)

    # Data to be prdicted
    data = [
        [
            "Yes you did. And you admitted to doing it. See the Warren Kinsella talk page."
        ],
        [
            "I asked you a question. We both know you have my page on your watch list, so are why are you playing games and making me formally ping you?  Makin'Bacon"
        ],
    ]

    index = 0
    run_states = multi_label_cls_task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    for result in results:
        # get predict index
        label_ids = []
        for i in range(dataset.num_labels):
            label_val = np.argmax(result[i])
            label_ids.append(label_val)
        print("%s\tpredict=%s" % (data[index][0], label_ids))
        index += 1
