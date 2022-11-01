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
from typing import List
from typing import Tuple

import paddle
import paddle2onnx
from easydict import EasyDict

from paddlehub.compat import paddle_utils
from paddlehub.compat.module import module_v1_utils
from paddlehub.utils import log
from paddlehub.utils import utils


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
            global_block.create_parameter(name=name,
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
        if not 'extra_info' in self.desc:
            return

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
    def context(self,
                signature: str = None,
                for_test: bool = False,
                trainable: bool = True,
                max_seq_len: int = 128) -> Tuple[dict, dict, paddle.static.Program]:
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

        # The bert series model saved by ModuleV1 sets max_seq_len to 512 by default. We need to adjust max_seq_len
        # according to the parameters in actual use.
        if 'bert' in self.name or self.name.startswith('ernie'):
            self._update_bert_max_seq_len(program, feed_dict, max_seq_len)

        return feed_dict, fetch_dict, program

    def _update_bert_max_seq_len(self, program: paddle.static.Program, feed_dict: dict, max_seq_len: int = 128):
        MAX_SEQ_LENGTH = 512
        if max_seq_len > MAX_SEQ_LENGTH or max_seq_len <= 0:
            raise ValueError("max_seq_len({}) should be in the range of [1, {}]".format(max_seq_len, MAX_SEQ_LENGTH))
        log.logger.info("Set maximum sequence length of input tensor to {}".format(max_seq_len))
        if self.name.startswith("ernie_v2"):
            feed_list = ["input_ids", "position_ids", "segment_ids", "input_mask", "task_ids"]
        else:
            feed_list = ["input_ids", "position_ids", "segment_ids", "input_mask"]
        for tensor_name in feed_list:
            seq_tensor_shape = [-1, max_seq_len, 1]
            log.logger.info("The shape of input tensor[{}] set to {}".format(tensor_name, seq_tensor_shape))
            program.global_block().var(feed_dict[tensor_name].name).desc.set_shape(seq_tensor_shape)

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

        # The naming of some old versions of Module is not standardized, which format of uppercase
        # letters. This will cause the path of these modules to be incorrect after installation.
        module_info = desc.module_info
        module_info.name = module_info.name.lower()
        return module_info

    def assets_path(self):
        return os.path.join(self.directory, 'assets')

    def get_name_prefix(self):
        return self.desc.name_prefix

    @property
    def is_runnable(self):
        '''
        Whether the Module is runnable, in other words, whether can we execute the Module through the
        `hub run` command.
        '''
        return self.default_signature != None

    @paddle_utils.run_in_static_mode
    def save_inference_model(self,
                             dirname: str,
                             model_filename: str = None,
                             params_filename: str = None,
                             combined: bool = False,
                             **kwargs):
        '''
        Export the model to Paddle Inference format.

        Args:
            dirname(str): The directory to save the paddle inference model.
            model_filename(str): The name of the saved model file. Default to `__model__`.
            params_filename(str): The name of the saved parameters file, only takes effect when `combined` is True.
                Default to `__params__`.
            combined(bool): Whether to save all parameters in a combined file. Default to True.
        '''
        if hasattr(self, 'processor'):
            if hasattr(self.processor, 'save_inference_model'):
                return self.processor.save_inference_model(dirname, model_filename, params_filename, combined)

        model_filename = '__model__' if not model_filename else model_filename
        if combined:
            params_filename = '__params__' if not params_filename else params_filename

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        feed_dict, fetch_dict, program = self.context(for_test=True, trainable=False)
        paddle.static.save_inference_model(dirname=dirname,
                                           main_program=program,
                                           executor=exe,
                                           feeded_var_names=[var.name for var in list(feed_dict.values())],
                                           target_vars=list(fetch_dict.values()),
                                           model_filename=model_filename,
                                           params_filename=params_filename)

        log.logger.info('Paddle Inference model saved in {}.'.format(dirname))

    @paddle_utils.run_in_static_mode
    def export_onnx_model(self, dirname: str, **kwargs):
        '''
        Export the model to ONNX format.

        Args:
            dirname(str): The directory to save the onnx model.
            **kwargs(dict|optional): Other export configuration options for compatibility, some may be removed
                in the future. Don't use them If not necessary. Refer to https://github.com/PaddlePaddle/paddle2onnx
                for more information.
        '''
        feed_dict, fetch_dict, program = self.context(for_test=True, trainable=False)
        inputs = set([var.name for var in feed_dict.values()])
        if self.type == 'CV/classification':
            outputs = [fetch_dict['class_probs']]
        else:
            outputs = set([var.name for var in fetch_dict.values()])
            outputs = [program.global_block().vars[key] for key in outputs]

        save_file = os.path.join(dirname, '{}.onnx'.format(self.name))
        paddle2onnx.program2onnx(program=program,
                                 scope=paddle.static.global_scope(),
                                 feed_var_names=inputs,
                                 target_vars=outputs,
                                 save_file=save_file,
                                 **kwargs)

    def sub_modules(self, recursive: bool = True):
        '''
        Get all sub modules.

        Args:
            recursive(bool): Whether to get sub modules recursively. Default to True.
        '''
        return []
