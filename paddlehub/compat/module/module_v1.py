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
from easydict import EasyDict

from paddlehub.compat import paddle_utils
from paddlehub.compat.module import module_v1_utils
from paddlehub.utils import utils, log


class ModuleV1(object):
    '''
    ModuleV1 is an old version of the PaddleHub Module format, which is no longer in use. In order to maintain
    compatibility, users can still load the corresponding Module for prediction. User should call `hub.Module`
    to initialize the corresponding object, rather than `ModuleV1`.
    '''

    # All ModuleV1 in PaddleHub is static graph model
    @paddle_utils.run_in_static_mode
    def __init__(self, name: str = None, directory: str = None, version: str = None):
        if not directory:
            return

        desc_file = os.path.join(directory, 'module_desc.pb')
        self.desc = module_v1_utils.convert_module_desc(desc_file)
        self.helper = self
        self.signatures = self.desc.signatures
        self.default_signature = self.desc.default_signature

        self.directory = directory
        self._load_model()
        self._load_parameters()
        self._load_processor()
        self._load_assets()
        self._load_extra_info()
        self._generate_func()

    def _load_processor(self):
        # Some module does not have a processor(e.g. ernie)
        if not 'processor_info' in self.desc:
            return

        python_path = os.path.join(self.directory, 'python')
        processor_name = self.desc.processor_info
        self.processor = utils.load_py_module(python_path, processor_name)
        self.processor = self.processor.Processor(module=self)

    def _load_assets(self):
        self.assets = []
        for file in os.listdir(self.assets_path()):
            filepath = os.path.join(self.assets_path(), file)
            self.assets.append(filepath)

    def _load_parameters(self):
        global_block = self.program.global_block()

        # record num parameters loaded by PaddleHub
        num_param_loaded = 0

        for param, attrs in self.desc.param_attrs.items():
            name = self.desc.name_prefix + param
            if not name in global_block.vars:
                continue

            num_param_loaded += 1
            var = global_block.vars[name]

            # Since the pre-trained model saved by the old version of Paddle cannot restore the corresponding
            # parameters, we need to restore them manually.
            global_block.create_parameter(
                name=name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                error_clip=var.error_clip,
                stop_gradient=var.stop_gradient,
                is_data=var.is_data,
                **attrs)

        log.logger.info('{} pretrained paramaters loaded by PaddleHub'.format(num_param_loaded))

    def _load_extra_info(self):
        for key, value in self.desc.extra_info.items():
            self.__dict__['get_{}'.format(key)] = value

    def _generate_func(self):
        for signature in self.desc.signatures:
            self.__dict__[signature] = functools.partial(self.__call__, sign_name=signature)

    def _load_model(self):
        model_path = os.path.join(self.directory, 'model')
        exe = paddle.static.Executor(paddle.CPUPlace())
        self.program, _, _ = paddle.static.load_inference_model(model_path, executor=exe)

        # Clear the callstack since it may leak the privacy of the creator.
        for block in self.program.blocks:
            for op in block.ops:
                if not 'op_callstack' in op.all_attrs():
                    continue
                op._set_attr('op_callstack', [''])

    @paddle_utils.run_in_static_mode
    def context(self, signature: str = None, for_test: bool = False,
                trainable: bool = True) -> Tuple[dict, dict, paddle.static.Program]:
        '''Get module context information, including graph structure and graph input and output variables.'''
        program = self.program.clone(for_test=for_test)
        paddle_utils.remove_feed_fetch_op(program)

        # generate feed vars and fetch vars from signatures
        feed_dict = {}
        fetch_dict = {}
        varinfos = [self.desc.signatures[signature]] if signature else self.desc.signatures.values()

        for info in varinfos:
            for feed_var in info.inputs:
                paddle_var = program.global_block().vars[feed_var.name]
                feed_dict[feed_var.alias] = paddle_var

            for fetch_var in info.outputs:
                paddle_var = program.global_block().vars[fetch_var.name]
                fetch_dict[fetch_var.alias] = paddle_var

        for param in program.all_parameters():
            param.trainable = trainable

        return feed_dict, fetch_dict, program

    @paddle_utils.run_in_static_mode
    def __call__(self, sign_name: str, data: dict, use_gpu: bool = False, batch_size: int = 1, **kwargs):
        '''Call the specified signature function for prediction.'''

        def _get_reader_and_feeder(data_format, data, place):
            def _reader(process_data):
                for item in zip(*process_data):
                    yield item

            process_data = []
            feed_name_list = []
            for key in data_format:
                process_data.append([value['processed'] for value in data[key]])
                feed_name_list.append(data_format[key]['feed_key'])
            feeder = paddle.fluid.DataFeeder(feed_list=feed_name_list, place=place)
            return functools.partial(_reader, process_data=process_data), feeder

        _, fetch_dict, program = self.context(signature=sign_name, for_test=True)
        fetch_list = list([value for key, value in fetch_dict.items()])
        with paddle.static.program_guard(program):
            result = []
            index = 0
            place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()

            exe = paddle.static.Executor(place=place)
            data = self.processor.preprocess(sign_name=sign_name, data_dict=data)
            data_format = self.processor.data_format(sign_name=sign_name)
            reader, feeder = _get_reader_and_feeder(data_format, data, place)
            reader = paddle.batch(reader, batch_size=batch_size)
            for batch in reader():
                data_out = exe.run(feed=feeder.feed(batch), fetch_list=fetch_list, return_numpy=False)
                sub_data = {key: value[index:index + len(batch)] for key, value in data.items()}
                result += self.processor.postprocess(sign_name, data_out, sub_data, **kwargs)
                index += len(batch)

        return result

    @classmethod
    def get_py_requirements(cls) -> List[str]:
        '''Get Module's python package dependency list.'''
        return []

    @classmethod
    def load(cls, directory: str) -> EasyDict:
        '''Load the Module object defined in the specified directory.'''
        module_info = cls.load_module_info(directory)

        # Generate a uuid based on the class information, and dynamically create a new type.
        # If we do not do this, the information generated later will overwrite the information
        # previously generated.
        cls_uuid = utils.md5(module_info.name + module_info.author + module_info.author_email + module_info.type +
                             module_info.summary + module_info.version + directory)
        cls = type('ModuleV1_{}'.format(cls_uuid), (cls, ), {})

        cls.name = module_info.name
        cls.author = module_info.author
        cls.author_email = module_info.author_email
        cls.type = module_info.type
        cls.summary = module_info.summary
        cls.version = utils.Version(module_info.version)
        cls.directory = directory
        return cls

    @classmethod
    def load_module_info(cls, directory: str) -> EasyDict:
        '''Load the infomation of Module object defined in the specified directory.'''
        desc_file = os.path.join(directory, 'module_desc.pb')
        desc = module_v1_utils.convert_module_desc(desc_file)
        return desc.module_info

    def assets_path(self):
        return os.path.join(self.directory, 'assets')

    @property
    def is_runnable(self):
        '''
        Whether the Module is runnable, in other words, whether can we execute the Module through the
        `hub run` command.
        '''
        return self.default_signature != None
