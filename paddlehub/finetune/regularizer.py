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

import os

import paddle.fluid as fluid
import numpy as np

import paddlehub as hub


class L2SPDecayRegularizer(fluid.regularizer.WeightDecayRegularizer):
    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L2SPDecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff
        self.save_dir = os.path.join(hub.CACHE_HOME, "l2sp")

    def __call__(self, param, grad, block):
        assert isinstance(param, fluid.framework.Parameter)
        assert isinstance(block, fluid.framework.Block)
        decay = block.create_var(
            name=fluid.unique_name.generate("l2sp_decay"),
            dtype=param.dtype,
            shape=param.shape,
            lod_level=param.lod_level)
        startpoint = block.create_var(
            name=fluid.unique_name.generate("l2sp_startpoint"),
            dtype=param.dtype,
            shape=param.shape,
            lod_level=param.lod_level)

        # TODO:record the start point with a more effective way
        # save startpoint
        save_program = fluid.default_startup_program()
        file_path = os.path.join(self.save_dir, param.name)
        with fluid.program_guard(save_program):
            save_block = save_program.global_block()
            save_var = save_block.create_var(
                name=param.name,
                shape=param.shape,
                dtype=param.dtype,
                type=param.type,
                lod_level=param.lod_level,
                persistable=True)
            save_block.append_op(
                type='save',
                inputs={'X': [save_var]},
                outputs={},
                attrs={'file_path': file_path})

        # load startpoint from file
        block.append_op(
            type='load',
            inputs={},
            outputs={'Out': [startpoint]},
            attrs={'file_path': file_path})

        # Append Op to calculate decay
        block.append_op(
            type='elementwise_sub',
            inputs={
                'X': param,
                'Y': startpoint
            },
            outputs={'Out': decay})
        block.append_op(
            type='scale',
            inputs={"X": decay},
            outputs={"Out": decay},
            attrs={"scale": self._regularization_coeff})

        return decay

    def __str__(self):
        return "L2SPDecay, regularization_coeff=%f" % self._regularization_coeff
