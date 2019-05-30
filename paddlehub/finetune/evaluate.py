#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import paddle.fluid as fluid
import numpy as np

from paddlehub.common.logger import logger
import paddlehub as hub


# Sequence label evaluation functions
def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):
    def extract_bio_chunk(seq):
        chunks = []
        cur_chunk = None
        null_index = tag_num - 1
        for index in range(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

            else:
                if cur_chunk is None:
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in range(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in range(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0
            while label_index < len(label_chunks) \
                   and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] \
                    < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] \
                    > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    if infer_chunks[infer_index]["en"] \
                        == label_chunks[label_index]["en"] \
                        and infer_chunks[infer_index]["type"] \
                        == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct


def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
