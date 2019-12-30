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

import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub

hub.common.logger.logger.setLevel("INFO")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=384, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub BERT pretrained model
    module = hub.Module(name="bert_uncased_L-12_H-768_A-12")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use ReadingComprehensionReader to read dataset
    # If you wanna load SQuAD 2.0 dataset, just set version_2_with_negative as True
    dataset = hub.dataset.SQUAD(version_2_with_negative=False)
    # dataset = hub.dataset.SQUAD(version_2_with_negative=True)

    reader = hub.reader.ReadingComprehensionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        doc_stride=128,
        max_query_length=64)

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
        warmup_proportion=args.warmup_proportion)

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        eval_interval=300,
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a reading comprehension finetune task by PaddleHub's API
    reading_comprehension_task = hub.ReadingComprehensionTask(
        data_reader=reader,
        feature=seq_output,
        feed_list=feed_list,
        config=config,
        sub_task="squad",
    )

    # Finetune by PaddleHub's API
    reading_comprehension_task.finetune_and_eval()
