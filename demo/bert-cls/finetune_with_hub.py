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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle_hub as hub

import reader.cls as reader
from utils.args import ArgumentGroup, print_arguments
from paddle_hub.finetune.config import FinetuneConfig

# yapf: disable
parser = argparse.ArgumentParser(__doc__)

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    3,       "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",     float,  5e-5,    "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay", "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float,  0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir",      str,  None,  "Path to training data.")
data_g.add_arg("vocab_path",    str,  None,  "Vocabulary path.")
data_g.add_arg("max_seq_len",   int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",    int,  32,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens",     bool, False,
              "If set, the batch size will be the maximum number of tokens in one batch. "
              "Otherwise, it will be the maximum number of examples in one batch.")

args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    print_arguments(args)
    config = FinetuneConfig(
        log_interval=10,
        eval_interval=100,
        save_ckpt_interval=200,
        use_cuda=True,
        checkpoint_dir="./bert_cls_ckpt",
        learning_rate=args.learning_rate,
        num_epoch=args.epoch,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        weight_decay=args.weight_decay,
        finetune_strategy="bert_finetune",
        with_memory_optimization=True,
        in_tokens=False,
        optimizer=None,
        warmup_proportion=args.warmup_proportion)

    # loading paddlehub BERT
    # module = hub.Module(
    #     module_dir="./hub_module/chinese_L-12_H-768_A-12.hub_module")
    module = hub.Module(module_dir="./hub_module/ernie-stable.hub_module")

    processor = reader.BERTClassifyReader(
        data_dir=args.data_dir,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    num_labels = len(processor.get_labels())

    # bert's input tensor, output tensor and forward graph
    # If you want to fine-tune the pretrain model parameter, please set
    # trainable to True
    input_dict, output_dict, train_program = module.context(
        sign_name="pooled_output", trainable=True)

    with fluid.program_guard(train_program):
        label = fluid.layers.data(name="label", shape=[1], dtype='int64')

        pooled_output = output_dict["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of bert's module need
        feed_list = [
            input_dict["src_ids"].name, input_dict["pos_ids"].name,
            input_dict["sent_ids"].name, input_dict["input_mask"].name,
            label.name
        ]
        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.append_mlp_classifier(
            pooled_output, label, num_classes=num_labels)

        # Finetune and evaluate by PaddleHub's API
        # will finish training, evaluation, testing, save model automatically
        hub.finetune_and_eval(
            task=cls_task,
            data_processor=processor,
            feed_list=feed_list,
            config=config)
