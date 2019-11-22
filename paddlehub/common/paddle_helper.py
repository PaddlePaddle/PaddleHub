#coding:utf-8
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import paddle
import paddle.fluid as fluid

from paddlehub.module import module_desc_pb2
from paddlehub.common.utils import from_pyobj_to_module_attr, from_module_attr_to_pyobj
from paddlehub.common.logger import logger

dtype_map = {
    fluid.core.VarDesc.VarType.FP32: "float32",
    fluid.core.VarDesc.VarType.FP64: "float64",
    fluid.core.VarDesc.VarType.FP16: "float16",
    fluid.core.VarDesc.VarType.INT32: "int32",
    fluid.core.VarDesc.VarType.INT16: "int16",
    fluid.core.VarDesc.VarType.INT64: "int64",
    fluid.core.VarDesc.VarType.BOOL: "bool",
    fluid.core.VarDesc.VarType.INT16: "int16",
    fluid.core.VarDesc.VarType.UINT8: "uint8",
    fluid.core.VarDesc.VarType.INT8: "int8",
}


def convert_dtype_to_string(dtype):
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise TypeError("dtype shoule in %s" % list(dtype_map.keys()))


def get_variable_info(var):
    if not isinstance(var, fluid.framework.Variable):
        raise TypeError("var shoule be an instance of fluid.framework.Variable")

    var_info = {
        'name': var.name,
        'dtype': convert_dtype_to_string(var.dtype),
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


def from_param_to_module_attr(param, module_attr):
    def paddle_obj_filter(pyobj):
        return isinstance(pyobj, fluid.framework.Variable) or isinstance(
            pyobj, fluid.framework.Block) or isinstance(
                pyobj, fluid.framework.Program) or isinstance(
                    pyobj, fluid.framework.Operator)

    module_attr.type = module_desc_pb2.MAP
    from_pyobj_to_module_attr(param.trainable,
                              module_attr.map.data['trainable'])
    from_pyobj_to_module_attr(param.do_model_average,
                              module_attr.map.data['do_model_average'])
    from_pyobj_to_module_attr(param.optimize_attr,
                              module_attr.map.data['optimize_attr'])
    from_pyobj_to_module_attr(
        param.regularizer,
        module_attr.map.data['regularizer'],
        obj_filter=paddle_obj_filter)
    from_pyobj_to_module_attr(
        param.gradient_clip_attr,
        module_attr.map.data['gradient_clip_attr'],
        obj_filter=paddle_obj_filter)


def from_module_attr_to_param(module_attr):
    param = {'gradient_clip_attr': None, 'regularizer': None}
    param['trainable'] = from_module_attr_to_pyobj(
        module_attr.map.data['trainable'])
    param['do_model_average'] = from_module_attr_to_pyobj(
        module_attr.map.data['do_model_average'])
    # do not recover learning rate
    #param['optimize_attr'] = from_module_attr_to_pyobj(
    #    module_attr.map.data['optimize_attr'])
    if module_attr.map.data['regularizer'].type != module_desc_pb2.NONE:
        regularizer_type = module_attr.map.data['regularizer'].name
        regularization_coeff = from_module_attr_to_pyobj(
            module_attr.map.data['regularizer'].object.
            data['_regularization_coeff'])
        param['regularizer'] = eval(
            "fluid.regularizer.%s(regularization_coeff = %f)" %
            (regularizer_type, regularization_coeff))

    if module_attr.map.data['gradient_clip_attr'].type != module_desc_pb2.NONE:
        clip_type = module_attr.map.data['gradient_clip_attr'].name
        if clip_type == "ErrorClipByValue" or clip_type == "GradientClipByValue":
            max = from_module_attr_to_pyobj(
                module_attr.map.data['gradient_clip_attr'].object.data['max'])
            min = from_module_attr_to_pyobj(
                module_attr.map.data['gradient_clip_attr'].object.data['min'])
            param['gradient_clip_attr'] = eval(
                "fluid.clip.%s(max = %f, min = %f)" % (clip_type, max, min))
        if clip_type == "GradientClipByNorm":
            clip_norm = from_module_attr_to_pyobj(
                module_attr.map.data['gradient_clip_attr'].object.
                data['clip_norm'])
            param['gradient_clip_attr'] = eval(
                "fluid.clip.%s(clip_norm = %f)" % (clip_type, clip_norm))
        if clip_type == "GradientClipByGlobalNorm":
            clip_norm = from_module_attr_to_pyobj(
                module_attr.map.data['gradient_clip_attr'].object.
                data['clip_norm'])
            group_name = from_module_attr_to_pyobj(
                module_attr.map.data['gradient_clip_attr'].object.
                data['group_name'])
            param['gradient_clip_attr'] = eval(
                "fluid.clip.%s(clip_norm = %f, group_name = \"%s\")" %
                (clip_type, clip_norm, group_name))

    return param


def _copy_vars_and_ops_in_blocks(from_block, to_block):
    for var in from_block.vars:
        var = from_block.var(var)
        var_info = copy.deepcopy(get_variable_info(var))
        if isinstance(var, fluid.framework.Parameter):
            to_block.create_parameter(**var_info)
        else:
            to_block.create_var(**var_info)

    for op in from_block.ops:
        op_info = {
            'type': op.type,
            'inputs': {
                input: [to_block.var(var) for var in op.input(input)]
                for input in op.input_names
            },
            'outputs': {
                output: [to_block.var(var) for var in op.output(output)]
                for output in op.output_names
            },
            'attrs': copy.deepcopy(op.all_attrs())
        }
        to_block.append_op(**op_info)


def connect_program(pre_program,
                    next_program,
                    input_dict=None,
                    inplace=True,
                    need_log=True):

    if not isinstance(pre_program, fluid.Program):
        raise TypeError("pre_program shoule be an instance of fluid.Program")

    if not isinstance(next_program, fluid.Program):
        raise TypeError("next_program shoule be an instance of fluid.Program")

    output_program = pre_program if inplace else pre_program.clone(
        for_test=False)
    if input_dict:
        if not isinstance(input_dict, dict):
            raise TypeError(
                "input_dict shoule be a python dict like {str:fluid.framework.Variable}"
            )

        for key, var in input_dict.items():
            if not isinstance(var, fluid.framework.Variable):
                raise TypeError(
                    "input_dict shoule be a python dict like {str:fluid.framework.Variable}"
                )

            var_info = copy.deepcopy(get_variable_info(var))
            input_var = output_program.global_block().create_var(**var_info)
            output_var = next_program.global_block().var(key)
            var_info = copy.deepcopy(get_variable_info(output_var))
            output_var = output_program.global_block().create_var(**var_info)
            output_program.global_block().append_op(
                type="assign",
                inputs={'X': input_var},
                outputs={'Out': output_var})

    block_map = {0: 0}
    if need_log:
        logger.info("Connect program's input tensor")
    for index, block in enumerate(next_program.blocks):
        if block.idx == 0:
            _copy_vars_and_ops_in_blocks(block, output_program.global_block())
        else:
            block_map[index] = len(output_program.blocks)
            logger.info(
                "block_%d in next_program merge into block_%d in pre_program" %
                (index, block_map[index]))
            new_block = output_program._create_block(
                parent_idx=block_map[block.parent_idx])
            _copy_vars_and_ops_in_blocks(block, new_block)
    if need_log:
        logger.info("Connect program's input tensor done")
    return output_program


def remove_feed_fetch_op(program):
    """ remove feed and fetch operator and variable for fine-tuning
    """
    block = program.global_block()
    need_to_remove_op_index = []
    for i, op in enumerate(block.ops):
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)

    for index in need_to_remove_op_index[::-1]:
        block._remove_op(index)

    need_to_remove_var = []
    for var in block.vars:
        if var.endswith("feed"):
            need_to_remove_var.append(var)
        if var.endswith("fetch"):
            need_to_remove_var.append(var)

    for var in need_to_remove_var:
        block._remove_var(var)

    program.desc.flush()


def set_parameter_trainable(program, trainable=True):
    for param in program.global_block().iter_parameters():
        param.trainable = trainable


def set_parameter_regularizer(program, regularizer):
    for param in program.global_block().iter_parameters():
        param.regularizer = regularizer


def set_parameter_learning_rate(program, learning_rate):
    for param in program.global_block().iter_parameters():
        param.optimize_attr['learning_rate'] = learning_rate


def set_op_attr(program, is_test=False):
    for block in program.blocks:
        for op in block.ops:
            if op.has_attr("is_test"):
                op._set_attr("is_test", is_test)


def clone_program(origin_program, for_test=False):
    dest_program = fluid.Program()
    _copy_vars_and_ops_in_blocks(origin_program.global_block(),
                                 dest_program.global_block())
    dest_program = dest_program.clone(for_test=for_test)
    if not for_test:
        for name, var in origin_program.global_block().vars.items():
            dest_program.global_block(
            ).vars[name].stop_gradient = var.stop_gradient

    return dest_program
