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
from paddlehub.tokenizer.bert_tokenizer import BertTokenizer

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use accuracy as metrics
    # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
    tokenizer = BertTokenizer(vocab_file=module.get_vocab_path())
    dataset = hub.dataset.BQ(tokenizer=tokenizer, max_length=args.max_seq_len)

    # For ernie_tiny, it use sub-word to tokenize chinese sentence
    # If not ernie tiny, sp_model_path and word_dict_path should be set None
    # reader = hub.reader.ClassifyReader(
    #     dataset=dataset,
    #     vocab_path=module.get_vocab_path(),
    #     max_seq_len=args.max_seq_len,
    #     sp_model_path=module.get_spm_path(),
    #     word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    # pooled_output =

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    # feed_list = [
    #     inputs["input_ids"].name,
    #     inputs["position_ids"].name,
    #     inputs["segment_ids"].name,
    #     inputs["input_mask"].name,
    # ]
    # print(feed_list)
    # exit()

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Define a classfication fine-tune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        dataset=dataset,
        feature=outputs["pooled_output"],
        # feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config)

    # Data to be prdicted
    text_a = [
        "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般", "交通方便；环境很好；服务态度很好 房间较小",
        "19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"
    ]
    # enconded_data = tokenizer.encode(text_a)
    enconded_data = list(map(tokenizer.encode, text_a))
    # print(enconded_data)
    # cls_task.finetune_and_eval()
    # cls_task.finetune_an
    print(
        cls_task.predict(
            data=enconded_data,
            label_list=dataset.get_labels(),
            accelerate_mode=False))
    # print(cls_task.predict(data=dataset.get_train_features()))
