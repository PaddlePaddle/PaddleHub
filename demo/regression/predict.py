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
"""Fine-tuning on classification task """

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

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE 2.0 pretrained model
    module = hub.Module(name="ernie_v2_eng_base")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use RegressionReader to read dataset
    dataset = hub.dataset.GLUE("STS-B")
    reader = hub.reader.RegressionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of ERNIE's module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Define a regression fine-tune task by PaddleHub's API
    reg_task = hub.RegressionTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        config=config,
    )

    # Data to be prdicted
    data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()[:10]]

    print(reg_task.predict(data=data, return_result=True))
