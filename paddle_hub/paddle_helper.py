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
from paddle_hub import module_desc_pb2
from paddle_hub.utils import from_pyobj_to_flexible_data, from_flexible_data_to_pyobj
from paddle_hub.logger import logger
import paddle
import paddle.fluid as fluid


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


def from_param_to_flexible_data(param, flexible_data):
    def paddle_obj_filter(pyobj):
        return isinstance(pyobj, fluid.framework.Variable) or isinstance(
            pyobj, fluid.framework.Block) or isinstance(
                pyobj, fluid.framework.Program) or isinstance(
                    pyobj, fluid.framework.Operator)

    flexible_data.type = module_desc_pb2.MAP
    from_pyobj_to_flexible_data(param.trainable,
                                flexible_data.map.data['trainable'])
    from_pyobj_to_flexible_data(param.do_model_average,
                                flexible_data.map.data['do_model_average'])
    from_pyobj_to_flexible_data(param.optimize_attr,
                                flexible_data.map.data['optimize_attr'])
    from_pyobj_to_flexible_data(
        param.regularizer,
        flexible_data.map.data['regularizer'],
        obj_filter=paddle_obj_filter)
    from_pyobj_to_flexible_data(
        param.gradient_clip_attr,
        flexible_data.map.data['gradient_clip_attr'],
        obj_filter=paddle_obj_filter)


def from_flexible_data_to_param(flexible_data):
    param = {'gradient_clip_attr': None, 'regularizer': None}
    param['trainable'] = from_flexible_data_to_pyobj(
        flexible_data.map.data['trainable'])
    param['do_model_average'] = from_flexible_data_to_pyobj(
        flexible_data.map.data['do_model_average'])
    param['optimize_attr'] = from_flexible_data_to_pyobj(
        flexible_data.map.data['optimize_attr'])
    if flexible_data.map.data['regularizer'].type != module_desc_pb2.NONE:
        regularizer_type = flexible_data.map.data['regularizer'].name
        regularization_coeff = from_flexible_data_to_pyobj(
            flexible_data.map.data['regularizer'].object.
            data['_regularization_coeff'])
        param['regularizer'] = eval(
            "fluid.regularizer.%s(regularization_coeff = %f)" %
            (regularizer_type, regularization_coeff))

    if flexible_data.map.data['gradient_clip_attr'].type != module_desc_pb2.NONE:
        clip_type = flexible_data.map.data['gradient_clip_attr'].name
        if clip_type == "ErrorClipByValue" or clip_type == "GradientClipByValue":
            max = from_flexible_data_to_pyobj(
                flexible_data.map.data['gradient_clip_attr'].object.data['max'])
            min = from_flexible_data_to_pyobj(
                flexible_data.map.data['gradient_clip_attr'].object.data['min'])
            param['gradient_clip_attr'] = eval(
                "fluid.clip.%s(max = %f, min = %f)" % (clip_type, max, min))
        if clip_type == "GradientClipByNorm":
            clip_norm = from_flexible_data_to_pyobj(
                flexible_data.map.data['gradient_clip_attr'].object.
                data['clip_norm'])
            param['gradient_clip_attr'] = eval(
                "fluid.clip.%s(clip_norm = %f)" % (clip_type, clip_norm))
        if clip_type == "GradientClipByGlobalNorm":
            clip_norm = from_flexible_data_to_pyobj(
                flexible_data.map.data['gradient_clip_attr'].object.
                data['clip_norm'])
            group_name = from_flexible_data_to_pyobj(
                flexible_data.map.data['gradient_clip_attr'].object.
                data['group_name'])
            param['gradient_clip_attr'] = eval(
                "fluid.clip.%s(clip_norm = %f, group_name = \"%s\")" %
                (clip_type, clip_norm, group_name))

    return param
