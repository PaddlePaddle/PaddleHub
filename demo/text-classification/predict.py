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
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(max_seq_len=args.max_seq_len)

    # Sentence classification  dataset reader
    dataset = hub.dataset.ChnSentiCorp()
    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

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

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=args.use_pyreader,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config)

    # Data to be prdicted
    data = [
        ["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
        [
            "还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。"
        ],
        [
            "前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级设施，下次不会再选择入住此店啦"
        ], ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]
    ]

    index = 0
    run_states = cls_task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    for batch_result in results:
        # get predict index
        batch_result = np.argmax(batch_result, axis=2)[0]
        for result in batch_result:
            print("%s\tpredict=%s" % (data[index][0], result))
            index += 1
