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
import contextlib
import copy
from typing import Callable
from typing import List

import paddle
from paddle.framework import core

from paddlehub.utils.utils import Version

dtype_map = {
    core.VarDesc.VarType.FP32: "float32",
    core.VarDesc.VarType.FP64: "float64",
    core.VarDesc.VarType.FP16: "float16",
    core.VarDesc.VarType.INT32: "int32",
    core.VarDesc.VarType.INT16: "int16",
    core.VarDesc.VarType.INT64: "int64",
    core.VarDesc.VarType.BOOL: "bool",
    core.VarDesc.VarType.INT16: "int16",
    core.VarDesc.VarType.UINT8: "uint8",
    core.VarDesc.VarType.INT8: "int8",
}


def convert_dtype_to_string(dtype: str) -> core.VarDesc.VarType:
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise TypeError("dtype shoule in %s" % list(dtype_map.keys()))


def get_variable_info(var: paddle.static.Variable) -> dict:
    if not isinstance(var, paddle.static.Variable):
        raise TypeError("var shoule be an instance of paddle.static.Variable")

    var_info = {
        'name': var.name,
        'stop_gradient': var.stop_gradient,
        'is_data': var.is_data,
        'error_clip': var.error_clip,
        'type': var.type
    }

    try:
        var_info['dtype'] = convert_dtype_to_string(var.dtype)
        var_info['lod_level'] = var.lod_level
        var_info['shape'] = var.shape
    except:
        pass

    if isinstance(var, paddle.device.framework.Parameter):
        var_info['trainable'] = var.trainable
        var_info['optimize_attr'] = var.optimize_attr
        var_info['regularizer'] = var.regularizer
        if Version(paddle.__version__) < '1.8':
            var_info['gradient_clip_attr'] = var.gradient_clip_attr
        var_info['do_model_average'] = var.do_model_average
    else:
        var_info['persistable'] = var.persistable

    return var_info


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


def rename_var(block: paddle.device.framework.Block, old_name: str, new_name: str):
    '''
    '''
    for op in block.ops:
        for input_name in op.input_arg_names:
            if input_name == old_name:
                op._rename_input(old_name, new_name)

        for output_name in op.output_arg_names:
            if output_name == old_name:
                op._rename_output(old_name, new_name)

    block._rename_var(old_name, new_name)


def add_vars_prefix(program: paddle.static.Program,
                    prefix: str,
                    vars: List[paddle.static.Variable] = None,
                    excludes: Callable = None):
    '''
    '''
    block = program.global_block()
    vars = list(vars) if vars else list(block.vars.keys())
    vars = [var for var in vars if var not in excludes] if excludes else vars
    for var in vars:
        rename_var(block, var, prefix + var)


def remove_vars_prefix(program: paddle.static.Program,
                       prefix: str,
                       vars: List[paddle.static.Variable] = None,
                       excludes: Callable = None):
    '''
    '''
    block = program.global_block()
    vars = [var for var in vars
            if var.startswith(prefix)] if vars else [var for var in block.vars.keys() if var.startswith(prefix)]
    vars = [var for var in vars if var not in excludes] if excludes else vars
    for var in vars:
        rename_var(block, var, var.replace(prefix, '', 1))


def clone_program(origin_program: paddle.static.Program, for_test: bool = False) -> paddle.static.Program:
    dest_program = paddle.static.Program()

    _copy_vars_and_ops_in_blocks(origin_program.global_block(), dest_program.global_block())

    dest_program = dest_program.clone(for_test=for_test)
    if not for_test:
        for name, var in origin_program.global_block().vars.items():
            dest_program.global_block().vars[name].stop_gradient = var.stop_gradient

    return dest_program


def _copy_vars_and_ops_in_blocks(from_block: paddle.device.framework.Block, to_block: paddle.device.framework.Block):
    for var in from_block.vars:
        var = from_block.var(var)
        var_info = copy.deepcopy(get_variable_info(var))
        if isinstance(var, paddle.device.framework.Parameter):
            to_block.create_parameter(**var_info)
        else:
            to_block.create_var(**var_info)

    for op in from_block.ops:
        all_attrs = op.all_attrs()
        if 'sub_block' in all_attrs:
            _sub_block = to_block.program._create_block()
            _copy_vars_and_ops_in_blocks(all_attrs['sub_block'], _sub_block)
            to_block.program._rollback()
            new_attrs = {'sub_block': _sub_block}
            for key, value in all_attrs.items():
                if key == 'sub_block':
                    continue
                new_attrs[key] = copy.deepcopy(value)
        else:
            new_attrs = copy.deepcopy(all_attrs)

        op_info = {
            'type': op.type,
            'inputs':
            {input: [to_block._find_var_recursive(var) for var in op.input(input)]
             for input in op.input_names},
            'outputs':
            {output: [to_block._find_var_recursive(var) for var in op.output(output)]
             for output in op.output_names},
            'attrs': new_attrs
        }
        to_block.append_op(**op_info)


def set_op_attr(program: paddle.static.Program, is_test: bool = False):
    for block in program.blocks:
        for op in block.ops:
            if not op.has_attr('is_test'):
                continue

            op._set_attr('is_test', is_test)


@contextlib.contextmanager
def static_mode_guard():
    '''enter static graph mode with `with` statement.'''
    premode = 'static' if not paddle.in_dynamic_mode() else 'dynamic'

    if premode == 'dynamic':
        paddle.enable_static()

    yield

    if premode == 'dynamic':
        paddle.disable_static()


def run_in_static_mode(func):
    '''Decorate a function to run in static graph mode.'''

    def runner(*args, **kwargs):
        with static_mode_guard():
            return func(*args, **kwargs)

    return runner
