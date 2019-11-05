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
import collections
import json
import io
import math
import numpy as np
import os
import six
import sys
import time

import paddle
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.finetune.task.reading_comprehension_task import write_predictions

hub.common.logger.logger.setLevel("INFO")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint.")
parser.add_argument("--result_dir", type=str, default=None, help="Directory to predicted results to be written.")
parser.add_argument("--max_seq_len", type=int, default=384, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
parser.add_argument("--max_answer_length", type=int, default=30, help="Max answer length.")
parser.add_argument("--n_best_size", type=int, default=20, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--null_score_diff_threshold", type=float, default=0.0, help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--dataset", type=str, default="squad", help="Support squad, squad2.0, drcd and cmrc2018")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Download dataset and use ReadingComprehensionReader to read dataset
    if args.dataset == "squad":
        dataset = hub.dataset.SQUAD(version_2_with_negative=False)
        module = hub.Module(name="bert_uncased_L-12_H-768_A-12")
    elif args.dataset == "squad2.0" or args.dataset == "squad2":
        args.dataset = "squad2.0"
        dataset = hub.dataset.SQUAD(version_2_with_negative=True)
        module = hub.Module(name="bert_uncased_L-12_H-768_A-12")
    elif args.dataset == "drcd":
        dataset = hub.dataset.DRCD()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
    elif args.dataset == "cmrc2018":
        dataset = hub.dataset.CMRC2018()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
    else:
        raise Exception(
            "Only support datasets: squad, squad2.0, drcd and cmrc2018")

    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    reader = hub.reader.ReadingComprehensionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        doc_stride=128,
        max_query_length=64)

    # Use "sequence_output" for token-level output.
    seq_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        lr_scheduler="linear_decay")

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        log_interval=10,
        use_pyreader=args.use_pyreader,
        use_data_parallel=args.use_data_parallel,
        save_ckpt_interval=100,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        enable_memory_optim=True,
        strategy=strategy)

    # Define a reading comprehension finetune task by PaddleHub's API
    reading_comprehension_task = hub.ReadingComprehensionTask(
        data_reader=reader,
        feature=seq_output,
        feed_list=feed_list,
        config=config)

    # Data to be predicted
    data = dataset.dev_examples[97:98]
    reading_comprehension_task.predict(data=data)
