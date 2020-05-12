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

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
parser.add_argument("--network", type=str, default='bilstm', help="Pre-defined network which was connected after Transformer model, such as ERNIE, BERT ,RoBERTa and ELECTRA.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use accuracy as metrics
    # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
    dataset = hub.dataset.ChnSentiCorp()

    # For ernie_tiny, it use sub-word to tokenize chinese sentence
    # If not ernie tiny, sp_model_path and word_dict_path should be set None
    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    token_feature = outputs["sequence_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Define a classfication finetune task by PaddleHub's API
    # network choice: bilstm, bow, cnn, dpcnn, gru, lstm (PaddleHub pre-defined network)
    # If you wanna add network after ERNIE/BERT/RoBERTa/ELECTRA module,
    # you must use the outputs["sequence_output"] as the token_feature of TextClassifierTask,
    # rather than outputs["pooled_output"], and feature is None
    cls_task = hub.TextClassifierTask(
        data_reader=reader,
        token_feature=token_feature,
        feed_list=feed_list,
        network=args.network,
        num_classes=dataset.num_labels,
        config=config)

    # Data to be prdicted
    data = [["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
            ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]]

    print(cls_task.predict(data=data, return_result=True))
