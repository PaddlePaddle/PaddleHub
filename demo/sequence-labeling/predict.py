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
"""Finetuning on sequence labeling task """

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
from paddlehub.finetune.evaluate import chunk_eval, calculate_f1

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--dataset", type=str, default="msra_ner", help="The choice of dataset")
parser.add_argument("--add_crf", type=ast.literal_eval, default=True, help="Whether use crf as decoder.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(max_seq_len=args.max_seq_len)

    # Download dataset and use SequenceLabelReader to read dataset
    if args.dataset.lower() == "msra_ner":
        dataset = hub.dataset.MSRA_NER()
    elif args.dataset.lower() == "express_ner":
        dataset = hub.dataset.Express_NER()
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)

    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)
    inv_label_map = {val: key for key, val in reader.label_map.items()}

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Construct transfer learning network
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

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

    # Define a sequence labeling finetune task by PaddleHub's API
    # if add crf, the network use crf as decoder
    seq_label_task = hub.SequenceLabelTask(
        data_reader=reader,
        feature=sequence_output,
        feed_list=feed_list,
        max_seq_len=args.max_seq_len,
        num_classes=dataset.num_labels,
        config=config,
        add_crf=args.add_crf)

    # test data
    data = [
        [u"黑龙江省双鸭山市尖山区八马路与东平行路交叉口北40米韦业涛18600009172"],
        [u"广西壮族自治区桂林市雁山区雁山镇西龙村老年活动中心17610348888羊卓卫"],
        [u"15652864561河南省开封市顺河回族区顺河区公园路32号赵本山"],
        [u"河北省唐山市玉田县无终大街159号18614253058尚汉生"],
        [u"台湾台中市北区北区锦新街18号18511226708蓟丽"],
    ]

    run_states = seq_label_task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]

    for num_batch, batch_results in enumerate(results):
        infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
        np_lens = batch_results[1]
        for index, np_len in enumerate(np_lens):
            labels = infers[index * args.max_seq_len:(index + 1) *
                            args.max_seq_len]
            label_str = ""
            sent_out_str = ""
            last_word = ""
            last_tag = ""
            #flag: cls position
            flag = 0
            count = 0
            for label_val in labels:
                label_tag = inv_label_map[label_val]
                if flag == 0:
                    flag = 1
                    continue
                cur_word = data[num_batch * args.batch_size + index][0][count]
                if last_word == "":
                    last_word = cur_word
                    last_tag = label_tag.split("-")[1]
                elif label_tag.startswith("B-"):
                    sent_out_str += last_word + u"/" + last_tag + u" "
                    last_word = data[num_batch * args.batch_size +
                                     index][0][count]
                    last_tag = label_tag.split("-")[1]
                elif label_tag.startswith("O"):
                    sent_out_str += last_word + u"/" + last_tag + u" "
                    last_word = data[num_batch * args.batch_size +
                                     index][0][count]
                    last_tag = label_tag
                elif label_tag.startswith("I-"):
                    last_word += cur_word
                else:
                    raise ValueError("invalid tag: %s" % (label_tag))
                count += 1
                if count == np_len - 1:
                    break
            if cur_word != "":
                sent_out_str += last_word + "/" + last_tag + " "
            print(sent_out_str)
