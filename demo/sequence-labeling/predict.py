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
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(max_seq_len=args.max_seq_len)

    # Sentence labeling dataset reader
    dataset = hub.dataset.MSRA_NER()
    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())
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
        add_crf=True)

    # test data
    data = [
        ["我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。"],
        ["为了跟踪国际最新食品工艺、流行趋势，大量搜集海外专业书刊资料是提高技艺的捷径。"],
        ["其中线装古籍逾千册；民国出版物几百种；珍本四册、稀见本四百余册，出版时间跨越三百余年。"],
        ["有的古木交柯，春机荣欣，从诗人句中得之，而入画中，观之令人心驰。"],
        ["不过重在晋趣，略增明人气息，妙在集古有道、不露痕迹罢了。"],
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
            count = 0
            for label_val in labels:
                label_str += inv_label_map[label_val]
                count += 1
                if count == np_len:
                    break

            # Drop the label results of CLS and SEP Token
            print(
                "%s\tpredict=%s" %
                (data[num_batch * args.batch_size + index][0], label_str[1:-1]))
