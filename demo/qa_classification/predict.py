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
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(max_seq_len=args.max_seq_len)

    # Download dataset and get its label list and label num
    # If you just want labels information, you can omit its tokenizer parameter to avoid preprocessing the train set.
    dataset = hub.dataset.NLPCC_DBQA()
    num_classes = dataset.num_labels
    label_list = dataset.get_labels()

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a classfication fine-tune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        dataset=dataset,
        feature=pooled_output,
        num_classes=dataset.num_labels,
        config=config)

    # Data to be prdicted
    data = [["北京奥运博物馆的场景效果负责人是谁？", "主要承担奥运文物征集、保管、研究和爱国主义教育基地建设相关工作。"],
            ["北京奥运博物馆的场景效果负责人是谁", "于海勃，美国加利福尼亚大学教授 场景效果负责人 总设计师"],
            ["北京奥运博物馆的场景效果负责人是谁？", "洪麦恩，清华大学美术学院教授 内容及主展线负责人 总设计师"]]
    # Use the appropriate tokenizer to preprocess the data
    # For ernie_tiny, it will do word segmentation to get subword. More details: https://www.jiqizhixin.com/articles/2019-11-06-9
    if module.name == "ernie_tiny":
        tokenizer = hub.ErnieTinyTokenizer(
            vocab_file=module.get_vocab_path(),
            spm_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path())
    else:
        tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())
    encoded_data = [
        tokenizer.encode(
            text=text, text_pair=text_pair, max_seq_len=args.max_seq_len)
        for text, text_pair in data
    ]
    print(cls_task.predict(data=encoded_data, label_list=label_list))
