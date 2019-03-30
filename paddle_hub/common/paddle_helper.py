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
from ..module import module_desc_pb2
from .utils import from_pyobj_to_flexible_data, from_flexible_data_to_pyobj
from .logger import logger
import paddle
import paddle.fluid as fluid
import copy


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
    #TODO(wuzewu): don't save learning rate
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
    # do not recover learning rate
    #param['optimize_attr'] = from_flexible_data_to_pyobj(
    #    flexible_data.map.data['optimize_attr'])
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


def connect_program(pre_program, next_program, input_dict=None, inplace=True):
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
                    input: [block.var(var) for var in op.input(input)]
                    for input in op.input_names
                },
                'outputs': {
                    output: [block.var(var) for var in op.output(output)]
                    for output in op.output_names
                },
                'attrs': copy.deepcopy(op.all_attrs())
            }
            to_block.append_op(**op_info)

    assert isinstance(pre_program,
                      fluid.Program), "pre_program should be fluid.Program"
    assert isinstance(next_program,
                      fluid.Program), "next_program should be fluid.Program"
    output_program = pre_program if inplace else pre_program.clone(
        for_test=False)
    if input_dict:
        assert isinstance(
            input_dict,
            dict), "the input_dict should be a dict with string-Variable pair"
        for key, var in input_dict.items():
            assert isinstance(
                var, fluid.framework.Variable
            ), "the input_dict should be a dict with string-Variable pair"
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
    logger.info("start to connect program")
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
    logger.info("end of connect program")
    return output_program


def remove_feed_fetch_op(program):
    """ remove feed and fetch operator and variable for fine-tuning
    """
    logger.info("remove feed fetch op")
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
