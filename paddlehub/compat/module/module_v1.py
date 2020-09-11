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

import functools
import os
from typing import Tuple, List

import paddle

from paddlehub.compat import paddle_utils
from paddlehub.compat.module import module_v1_utils
from paddlehub.utils import utils, log


class ModuleV1(object):
    '''
    '''
    def __init__(self, name: str = None, directory: str = None, version: str = None):
        if not directory:
            return

        self.directory = directory
        desc_file = os.path.join(directory, 'module_desc.pb')
        self.desc = module_v1_utils.convert_module_desc(desc_file)
        self._load_model()
        self._load_parameters()
        self._load_processor()
        self._load_assets()
        self._load_extra_info()
        self._load_signatures()

    def _load_processor(self):
        python_path = os.path.join(self.directory, 'python')
        processor_name = self.desc.processor_info
        self.processor = utils.load_py_module(python_path, processor_name)

    def _load_assets(self):
        assets_path = os.path.join(self.directory, 'assets')
        self.assets = []
        for file in os.listdir(assets_path):
            filepath = os.path.join(assets_path, file)
            self.assets.append(filepath)

    def _load_parameters(self):
        global_block = self.program.global_block()
        for param, attrs in self.desc.param_attrs.items():
            name = self.desc.name_prefix + param
            if not name in global_block.vars:
                continue

            var = global_block.vars[name]
            global_block.create_parameter(name=name,
                                          shape=var.shape,
                                          dtype=var.dtype,
                                          type=var.type,
                                          lod_level=var.lod_level,
                                          error_clip=var.error_clip,
                                          stop_gradient=var.stop_gradient,
                                          is_data=var.is_data,
                                          **attrs)

    def _load_extra_info(self):
        for key, value in self.desc.extra_info.items():
            self.__dict__['get_{}'.format(key)] = value

    def _load_signatures(self):
        for signature in self.desc.signatures:
            self.__dict__[signature] = functools.partial(self.__call__, signature=signature)

    def _load_model(self):
        model_path = os.path.join(self.directory, 'model')
        exe = paddle.static.Executor(paddle.CPUPlace())
        self.program, _, _ = paddle.io.load_inference_model(model_path, executor=exe)

        # Clear the callstack since it may leak the privacy of the creator.
        for block in self.program.blocks:
            for op in block.ops:
                if not 'op_callstack' in op.all_attrs():
                    continue
                op._set_attr('op_callstack', [''])

    def context(self, for_test: bool = False, trainable: bool = True) -> Tuple[dict, dict, paddle.static.Program]:
        '''
        '''
        program = self.program.clone(for_test=for_test)
        paddle_utils.remove_feed_fetch_op(program)

        # generate feed vars and fetch vars from signatures
        feed_dict = {}
        fetch_dict = {}
        for info in self.desc.signatures.values():
            for feed_var in info.feed_vars:
                paddle_var = program.global_block().vars[feed_var.name]
                feed_dict[feed_var.alias] = paddle_var

            for fetch_var in info.fetch_vars:
                paddle_var = program.global_block().vars[fetch_var.name]
                fetch_dict[fetch_var.alias] = paddle_var

        # record num parameters loaded by PaddleHub
        num_param_loaded = 0
        for param in program.all_parameters():
            num_param_loaded += 1
            param.trainable = trainable

        log.logger.info('{} pretrained paramaters loaded by PaddleHub'.format(num_param_loaded))

        return feed_dict, fetch_dict, program

    def __call__(self, signature, data, use_gpu: bool = False, batch_size: int = 1, **kwargs):
        '''
        '''
        ...

    @classmethod
    def get_py_requirements(cls) -> List[str]:
        return []

    @classmethod
    def load(cls, desc_file):
        desc = module_v1_utils.convert_module_desc(desc_file)

        cls.author = desc.module_info.author
        cls.author_email = desc.module_info.author_email
        cls.summary = desc.module_info.summary
        cls.type = desc.module_info.type
        cls.name = desc.module_info.name
        cls.version = utils.Version(desc.module_info.version)
        return cls
