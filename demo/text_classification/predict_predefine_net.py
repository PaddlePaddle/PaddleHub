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

import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
parser.add_argument("--network", type=str, default='bilstm', help="Pre-defined network which was connected after Transformer model, such as ERNIE, BERT ,RoBERTa and ELECTRA.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and get its label list and label num
    # If you just want labels information, you can omit its tokenizer parameter to avoid preprocessing the train set.
    dataset = hub.dataset.ChnSentiCorp()
    num_classes = dataset.num_labels
    label_list = dataset.get_labels()

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    token_feature = outputs["sequence_output"]

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Define a classfication fine-tune task by PaddleHub's API
    # network choice: bilstm, bow, cnn, dpcnn, gru, lstm (PaddleHub pre-defined network)
    # If you wanna add network after ERNIE/BERT/RoBERTa/ELECTRA module,
    # you must use the outputs["sequence_output"] as the token_feature of TextClassifierTask,
    # rather than outputs["pooled_output"], and feature is None
    cls_task = hub.TextClassifierTask(
        token_feature=token_feature,
        network=args.network,
        num_classes=dataset.num_labels,
        config=config)

    # Data to be prdicted
    text_a = [
        "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般", "交通方便；环境很好；服务态度很好 房间较小",
        "19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"
    ]
    # Use the appropriate tokenizer to preprocess the data
    # For ernie_tiny, it will do word segmentation to get subword. More details: https://www.jiqizhixin.com/articles/2019-11-06-9
    if module.name == "ernie_tiny":
        tokenizer = hub.ErnieTinyTokenizer(
            vocab_file=module.get_vocab_path(),
            spm_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path(),
            max_seq_len=args.max_seq_len)
    else:
        tokenizer = hub.BertTokenizer(
            vocab_file=module.get_vocab_path(), max_seq_len=args.max_seq_len)
    encoded_data = [tokenizer.encode(text=text) for text in text_a]
    print(cls_task.predict(data=encoded_data, label_list=label_list))
