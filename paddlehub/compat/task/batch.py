# coding:utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
'''Mask, padding and batching.'''

from typing import List, Union

import numpy as np


def pad_batch_data(insts: List,
                   pad_idx: int = 0,
                   max_seq_len: int = 128,
                   return_pos: bool = False,
                   return_input_mask: bool = False,
                   return_max_len: bool = False,
                   return_num_token: bool = False,
                   return_seq_lens: bool = False) -> Union[List, np.ndarray]:
    '''
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and input mask.
    '''
    return_list = []
    #max_len = max(len(inst) for inst in insts)
    max_len = max_seq_len
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype('int64').reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype('int64').reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype('float32')]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype('int64').reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]
