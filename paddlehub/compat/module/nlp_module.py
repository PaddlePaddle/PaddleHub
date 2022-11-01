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
import argparse
import ast
import os
import re
from typing import Any
from typing import List
from typing import Text
from typing import Tuple

import numpy as np
import paddle
import six
from paddle.framework import core

from paddlehub.compat import paddle_utils
from paddlehub.compat.task.config import RunConfig
from paddlehub.compat.task.reader import ClassifyReader
from paddlehub.compat.task.transformer_emb_task import TransformerEmbeddingTask
from paddlehub.module.module import RunModule
from paddlehub.module.module import runnable
from paddlehub.utils.parser import txt_parser
from paddlehub.utils.utils import sys_stdin_encoding


class DataFormatError(Exception):

    def __init__(self, *args):
        self.args = args


class NLPBaseModule(RunModule):

    def get_vocab_path(self):
        '''
        Get the path to the vocabulary whih was used to pretrain
        Returns:
             self.vocab_path(str): the path to vocabulary
        '''
        return self.vocab_path


class NLPPredictionModule(NLPBaseModule):

    def _set_config(self):
        '''predictor config setting'''
        cpu_config = core.AnalysisConfig(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = core.create_paddle_predictor(cpu_config)

        try:
            _places = os.environ['CUDA_VISIBLE_DEVICES']
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = core.AnalysisConfig(self.pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = core.create_paddle_predictor(gpu_config)

    def texts2tensor(self, texts: List[dict]) -> paddle.Tensor:
        '''
        Tranform the texts(dict) to PaddleTensor
        Args:
             texts(list): each element is a dict that must have a named 'processed' key whose value is word_ids, such as
                          texts = [{'processed': [23, 89, 43, 906]}]
        Returns:
             tensor(PaddleTensor): tensor with texts data
        '''
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            data += text['processed']
            lod.append(len(text['processed']) + lod[i])
        tensor = core.PaddleTensor(np.array(data).astype('int64'))
        tensor.name = 'words'
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]
        return tensor

    def to_unicode(self, texts: str) -> Text:
        '''
        Convert each element's type(str) of texts(list) to unicode in python2.7
        Args:
             texts(list): each element's type is str in python2.7
        Returns:
             texts(list): each element's type is unicode in python2.7
        '''
        if six.PY2:
            unicode_texts = []
            for text in texts:
                if isinstance(text, six.string_types):
                    unicode_texts.append(text.decode(sys_stdin_encoding()).decode('utf8'))
                else:
                    unicode_texts.append(text)
            texts = unicode_texts
        return texts

    @runnable
    def run_cmd(self, argvs: List[Any]):
        '''Run as a command'''
        self.parser = argparse.ArgumentParser(description='Run the %s module.' % self.name,
                                              prog='hub run %s' % self.name,
                                              usage='%(prog)s',
                                              add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title='Input options', description='Input data. Required')
        self.arg_config_group = self.parser.add_argument_group(
            title='Config options', description='Run configuration for controlling module behavior, not required.')

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.predict(texts=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)

        return results

    def add_module_config_arg(self):
        '''Add the command config options'''
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help='whether use GPU for prediction')

        self.arg_config_group.add_argument('--batch_size', type=int, default=1, help='batch size for prediction')

    def add_module_input_arg(self):
        '''Add the command input options'''
        self.arg_input_group.add_argument('--input_file', type=str, default=None, help='file contain input data')
        self.arg_input_group.add_argument('--input_text', type=str, default=None, help='text to predict')

    def check_input_data(self, args):
        input_data = []
        if args.input_file:
            if not os.path.exists(args.input_file):
                raise FileNotFoundError('File %s does not exist.' % args.input_file)
            else:
                input_data = txt_parser.parse(args.input_file, use_strip=True)
        elif args.input_text:
            input_data = [args.input_text]

        return input_data


class TransformerModule(NLPBaseModule):
    '''
    Tranformer Module base class can be used by BERT, ERNIE, RoBERTa and so on.
    '''

    def __init__(self,
                 name: str = None,
                 directory: str = None,
                 module_dir: List = None,
                 version: str = None,
                 max_seq_len: int = 128,
                 **kwargs):
        if not directory:
            return
        super(TransformerModule, self).__init__(name=name,
                                                directory=directory,
                                                module_dir=module_dir,
                                                version=version,
                                                **kwargs)

        self.max_seq_len = max_seq_len

    def init_pretraining_params(self, exe: paddle.static.Executor, pretraining_params_path: str,
                                main_program: paddle.static.Program):
        assert os.path.exists(pretraining_params_path), '[{}] cann\'t be found.'.format(pretraining_params_path)

        def existed_params(var):
            if not isinstance(var, paddle.device.framework.Parameter):
                return False
            return os.path.exists(os.path.join(pretraining_params_path, var.name))

        paddle.static.load(executor=exe,
                           model_path=pretraining_params_path,
                           program=main_program,
                           var_list=main_program.all_parameters())

    def param_prefix(self) -> str:
        return '@HUB_%s@' % self.name

    @paddle_utils.run_in_static_mode
    def context(
        self,
        max_seq_len: int = None,
        trainable: bool = True,
        num_slots: int = 1,
    ) -> Tuple[dict, dict, paddle.static.Program]:
        '''
        get inputs, outputs and program from pre-trained module
        Args:
            max_seq_len (int): It will limit the total sequence returned so that it has a maximum length.
            trainable (bool): Whether fine-tune the pre-trained module parameters or not.
            num_slots(int): It's number of data inputted to the model, selectted as following options:
                - 1(default): There's only one data to be feeded in the model, e.g. the module is used for sentence classification task.
                - 2: There are two data to be feeded in the model, e.g. the module is used for text matching task (point-wise).
                - 3: There are three data to be feeded in the model, e.g. the module is used for text matching task (pair-wise).
        Returns: inputs, outputs, program.
                 The inputs is a dict with keys named input_ids, position_ids, segment_ids, input_mask and task_ids
                 The outputs is a dict with two keys named pooled_output and sequence_output.
        '''
        assert num_slots >= 1 and num_slots <= 3, 'num_slots must be 1, 2, or 3, but the input is %d' % num_slots
        if not max_seq_len:
            max_seq_len = self.max_seq_len

        assert max_seq_len <= self.MAX_SEQ_LEN and max_seq_len >= 1, 'max_seq_len({}) should be in the range of [1, {}]'.format(
            max_seq_len, self.MAX_SEQ_LEN)

        module_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(module_program, startup_program):
            with paddle.utils.unique_name.guar.guard():
                input_ids = paddle.static.data(name='input_ids', shape=[-1, max_seq_len, 1], dtype='int64', lod_level=0)
                position_ids = paddle.static.data(name='position_ids',
                                                  shape=[-1, max_seq_len, 1],
                                                  dtype='int64',
                                                  lod_level=0)
                segment_ids = paddle.static.data(name='segment_ids',
                                                 shape=[-1, max_seq_len, 1],
                                                 dtype='int64',
                                                 lod_level=0)
                input_mask = paddle.static.data(name='input_mask',
                                                shape=[-1, max_seq_len, 1],
                                                dtype='float32',
                                                lod_level=0)
                pooled_output, sequence_output = self.net(input_ids, position_ids, segment_ids, input_mask)

                data_list = [(input_ids, position_ids, segment_ids, input_mask)]
                output_name_list = [(pooled_output.name, sequence_output.name)]

                if num_slots > 1:
                    input_ids_2 = paddle.static.data(name='input_ids_2',
                                                     shape=[-1, max_seq_len, 1],
                                                     dtype='int64',
                                                     lod_level=0)
                    position_ids_2 = paddle.static.data(name='position_ids_2',
                                                        shape=[-1, max_seq_len, 1],
                                                        dtype='int64',
                                                        lod_level=0)
                    segment_ids_2 = paddle.static.data(name='segment_ids_2',
                                                       shape=[-1, max_seq_len, 1],
                                                       dtype='int64',
                                                       lod_level=0)
                    input_mask_2 = paddle.static.data(name='input_mask_2',
                                                      shape=[-1, max_seq_len, 1],
                                                      dtype='float32',
                                                      lod_level=0)
                    pooled_output_2, sequence_output_2 = self.net(input_ids_2, position_ids_2, segment_ids_2,
                                                                  input_mask_2)
                    data_list.append((input_ids_2, position_ids_2, segment_ids_2, input_mask_2))
                    output_name_list.append((pooled_output_2.name, sequence_output_2.name))

                if num_slots > 2:
                    input_ids_3 = paddle.static.data(name='input_ids_3',
                                                     shape=[-1, max_seq_len, 1],
                                                     dtype='int64',
                                                     lod_level=0)
                    position_ids_3 = paddle.static.data(name='position_ids_3',
                                                        shape=[-1, max_seq_len, 1],
                                                        dtype='int64',
                                                        lod_level=0)
                    segment_ids_3 = paddle.static.data(name='segment_ids_3',
                                                       shape=[-1, max_seq_len, 1],
                                                       dtype='int64',
                                                       lod_level=0)
                    input_mask_3 = paddle.static.data(name='input_mask_3',
                                                      shape=[-1, max_seq_len, 1],
                                                      dtype='float32',
                                                      lod_level=0)
                    pooled_output_3, sequence_output_3 = self.net(input_ids_3, position_ids_3, segment_ids_3,
                                                                  input_mask_3)
                    data_list.append((input_ids_3, position_ids_3, segment_ids_3, input_mask_3))
                    output_name_list.append((pooled_output_3.name, sequence_output_3.name))

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        # To be compatible with the module v1
        vars = filter(
            lambda var: var not in [
                'input_ids', 'position_ids', 'segment_ids', 'input_mask', 'input_ids_2', 'position_ids_2',
                'segment_ids_2', 'input_mask_2', 'input_ids_3', 'position_ids_3', 'segment_ids_3', 'input_mask_3'
            ], list(module_program.global_block().vars.keys()))
        paddle_utils.add_vars_prefix(program=module_program, prefix=self.param_prefix(), vars=vars)
        self.init_pretraining_params(exe, self.params_path, main_program=module_program)

        self.params_layer = {}
        for param in module_program.global_block().iter_parameters():
            param.trainable = trainable
            match = re.match(r'.*layer_(\d+).*', param.name)
            if match:
                # layer num begins from 0
                layer = match.group(1)
                self.params_layer[param.name] = int(layer)

        inputs = {}
        outputs = {}
        for index, data in enumerate(data_list):

            if index == 0:
                inputs['input_ids'] = data[0]
                inputs['position_ids'] = data[1]
                inputs['segment_ids'] = data[2]
                inputs['input_mask'] = data[3]
                outputs['pooled_output'] = module_program.global_block().vars[self.param_prefix() +
                                                                              output_name_list[0][0]]
                outputs['sequence_output'] = module_program.global_block().vars[self.param_prefix() +
                                                                                output_name_list[0][1]]
            else:
                inputs['input_ids_%s' % (index + 1)] = data[0]
                inputs['position_ids_%s' % (index + 1)] = data[1]
                inputs['segment_ids_%s' % (index + 1)] = data[2]
                inputs['input_mask_%s' % (index + 1)] = data[3]
                outputs['pooled_output_%s' %
                        (index + 1)] = module_program.global_block().vars[self.param_prefix() +
                                                                          output_name_list[index][0]]
                outputs['sequence_output_%s' %
                        (index + 1)] = module_program.global_block().vars[self.param_prefix() +
                                                                          output_name_list[index][1]]

        return inputs, outputs, module_program

    @paddle_utils.run_in_static_mode
    def get_embedding(self, texts: List[str], max_seq_len: int = 512, use_gpu: bool = False, batch_size: int = 1):
        '''
        get pooled_output and sequence_output for input texts.
        Warnings: this method depends on Paddle Inference Library, it may not work properly in PaddlePaddle <= 1.6.2.
        Args:
            texts (list): each element is a text sample, each sample include text_a and text_b where text_b can be omitted.
                          for example: [[sample0_text_a, sample0_text_b], [sample1_text_a, sample1_text_b], ...]
            max_seq_len (int): the max sequence length.
            use_gpu (bool): use gpu or not, default False.
            batch_size (int): the data batch size, default 1.
        Returns:
            pooled_outputs(list): its element is a numpy array, the first feature of each text sample.
            sequence_outputs(list): its element is a numpy array, the whole features of each text sample.
        '''
        if not hasattr(self,
                       'emb_job') or self.emb_job['batch_size'] != batch_size or self.emb_job['use_gpu'] != use_gpu:
            inputs, outputs, program = self.context(trainable=True, max_seq_len=max_seq_len)

            reader = ClassifyReader(
                dataset=None,
                vocab_path=self.get_vocab_path(),
                max_seq_len=max_seq_len,
                sp_model_path=self.get_spm_path() if hasattr(self, 'get_spm_path') else None,
                word_dict_path=self.get_word_dict_path() if hasattr(self, 'word_dict_path') else None)

            feed_list = [
                inputs['input_ids'].name,
                inputs['position_ids'].name,
                inputs['segment_ids'].name,
                inputs['input_mask'].name,
            ]

            pooled_feature, seq_feature = outputs['pooled_output'], outputs['sequence_output']

            config = RunConfig(use_data_parallel=False, use_cuda=use_gpu, batch_size=batch_size)

            self.emb_job = {}
            self.emb_job['task'] = TransformerEmbeddingTask(
                pooled_feature=pooled_feature,
                seq_feature=seq_feature,
                feed_list=feed_list,
                data_reader=reader,
                config=config,
            )
            self.emb_job['batch_size'] = batch_size
            self.emb_job['use_gpu'] = use_gpu

        return self.emb_job['task'].predict(data=texts, return_result=True, accelerate_mode=True)

    def get_spm_path(self) -> str:
        if hasattr(self, 'spm_path'):
            return self.spm_path
        return None

    def get_word_dict_path(self) -> str:
        if hasattr(self, 'word_dict_path'):
            return self.word_dict_path
        return None

    def get_params_layer(self) -> dict:
        if not hasattr(self, 'params_layer'):
            raise AttributeError('The module context has not been initialized. '
                                 'Please call context() before using get_params_layer')
        return self.params_layer
