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

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid


def to_list(input):
    if not isinstance(input, list):
        if not isinstance(input, tuple):
            input = [input]

    return input


def get_variable_info(var):
    assert isinstance(
        var,
        fluid.framework.Variable), "var should be a fluid.framework.Variable"
    var_info = {
        'type': var.type,
        'name': var.name,
        'dtype': var.dtype,
        'lod_level': var.lod_level,
        'shape': var.shape,
        'stop_gradient': var.stop_gradient,
        'is_data': var.is_data,
        'error_clip': var.error_clip
    }
    if isinstance(var, fluid.framework.Parameter):
        var_info['trainable'] = var.trainable
        var_info['optimize_attr'] = var.optimize_attr
        var_info['regularizer'] = var.regularizer
        var_info['gradient_clip_attr'] = var.gradient_clip_attr
        var_info['do_model_average'] = var.do_model_average
    else:
        var_info['persistable'] = var.persistable

    return var_info
