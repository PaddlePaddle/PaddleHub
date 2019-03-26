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
import collections
import time
import argparse
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle_hub as hub

import reader.cls as reader
from model.bert import BertConfig
from model.classifier import create_model_with_hub, create_model
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path",         str,  None,           "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",          str,  None,           "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params",  str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",              str,  "checkpoints",  "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    3,       "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",     float,  5e-5,    "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float,  0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("use_fp16",          bool,   False,   "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir",      str,  None,  "Path to training data.")
data_g.add_arg("vocab_path",    str,  None,  "Vocabulary path.")
data_g.add_arg("max_seq_len",   int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",    int,  32,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens",     bool, False,
              "If set, the batch size will be the maximum number of tokens in one batch. "
              "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed",   int,  0,     "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,     "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("task_name",                    str,    None,
                   "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")

args = parser.parse_args()
# yapf: enable.


def test_hub_api(args, config):

    processor = reader.ChnsenticorpProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed)

    num_labels = len(processor.get_labels())

    # loading paddlehub BERT
    module = hub.Module(module_dir="./chinese_L-12_H-768_A-12.hub_module")

    input_dict, output_dict, train_program = module.context(
        sign_name="pooled_output", trainable=True)

    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        label = fluid.layers.data(name="label", shape=[1], dtype='int64')

        pooled_output = output_dict["pooled_output"]

        # setup feed list for data feeder
        feed_list = [
            input_dict["src_ids"].name, input_dict["pos_ids"].name,
            input_dict["sent_ids"].name, input_dict["input_mask"].name,
            label.name
        ]
        task = hub.append_mlp_classifier(
            pooled_output, label, num_classes=num_labels)

        hub.finetune_and_eval(task, feed_list, processor, config)


FinetuneConfig = collections.namedtuple('FinetuneConfig', [
    'stat_interval', 'eval_interval', 'use_cuda', 'learning_rate',
    'weight_decay', 'in_tokens', 'epoch', 'batch_size', 'max_seq_len',
    'warmup_proportion'
])

if __name__ == '__main__':
    print_arguments(args)
    config = FinetuneConfig(
        stat_interval=10,
        eval_interval=100,
        use_cuda=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        in_tokens=args.in_tokens,
        epoch=args.epoch,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        warmup_proportion=args.warmup_proportion)
    test_hub_api(args, config)
