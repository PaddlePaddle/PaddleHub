# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def remove_feed_fetch_op(program: paddle.static.Program):
    '''Remove feed and fetch operator and variable for fine-tuning.'''
    block = program.global_block()
    need_to_remove_op_index = []

    for i, op in enumerate(block.ops):
        if op.type == 'feed' or op.type == "fetch":
            need_to_remove_op_index.append(i)

    for index in need_to_remove_op_index[::-1]:
        block._remove_op(index)

    need_to_remove_var = []
    for var in block.vars:
        if var.endswith("feed"):
            need_to_remove_var.append(var)
        if var.endswith('fetch'):
            need_to_remove_var.append(var)

    for var in need_to_remove_var:
        block._remove_var(var)

    program.desc.flush()
