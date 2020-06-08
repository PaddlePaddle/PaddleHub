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
"""Fine-tuning on sequence labeling task """

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
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(max_seq_len=args.max_seq_len)

    # Download dataset and get its label list and label num
    # If you just want labels information, you can omit its tokenizer parameter to avoid preprocessing the train set.
    dataset = hub.dataset.MSRA_NER()
    num_classes = dataset.num_labels
    label_list = dataset.get_labels()

    # Construct transfer learning network
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a sequence labeling fine-tune task by PaddleHub's API
    # if add crf, the network use crf as decoder
    seq_label_task = hub.SequenceLabelTask(
        feature=sequence_output,
        max_seq_len=args.max_seq_len,
        num_classes=num_classes,
        config=config,
        add_crf=False)

    # Data to be predicted
    # If using python 2, prefix "u" is necessary
    text_a = [
        "我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。",
        "为了跟踪国际最新食品工艺、流行趋势，大量搜集海外专业书刊资料是提高技艺的捷径。",
        "其中线装古籍逾千册；民国出版物几百种；珍本四册、稀见本四百余册，出版时间跨越三百余年。",
        "有的古木交柯，春机荣欣，从诗人句中得之，而入画中，观之令人心驰。",
        "不过重在晋趣，略增明人气息，妙在集古有道、不露痕迹罢了。",
    ]

    # Add 0x02 between characters to match the format of training data,
    # otherwise the length of prediction results will not match the input string
    # if the input string contains non-Chinese characters.
    formatted_text_a = list(map("\002".join, text_a))

    # Use the appropriate tokenizer to preprocess the data
    # For ernie_tiny, it use BertTokenizer too.
    tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())
    encoded_data = [
        tokenizer.encode(text=text, max_seq_len=args.max_seq_len)
        for text in formatted_text_a
    ]
    print(seq_label_task.predict(data=encoded_data, label_list=label_list))
