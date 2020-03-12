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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import json
import os
import re
import six

import numpy as np
import paddle.fluid as fluid
from paddlehub.common import paddle_helper
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import runnable


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


class NLPBaseModule(hub.Module):
    def _initialize(self):
        """
        initialize with the necessary elements
        This method must be overrided.
        """
        raise NotImplementedError()

    def get_vocab_path(self):
        """
        Get the path to the vocabulary whih was used to pretrain

        Returns:
             self.vocab_path(str): the path to vocabulary
        """
        return self.vocab_path


class NLPPredictionModule(NLPBaseModule):
    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_paddle_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = AnalysisConfig(self.pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def texts2tensor(self, texts):
        """
        Tranform the texts(dict) to PaddleTensor
        Args:
             texts(list): each element is a dict that must have a named 'processed' key whose value is word_ids, such as
                          texts = [{'processed': [23, 89, 43, 906]}]
        Returns:
             tensor(PaddleTensor): tensor with texts data
        """
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            data += text['processed']
            lod.append(len(text['processed']) + lod[i])
        tensor = PaddleTensor(np.array(data).astype('int64'))
        tensor.name = "words"
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]
        return tensor

    def to_unicode(self, texts):
        """
        Convert each element's type(str) of texts(list) to unicode in python2.7
        Args:
             texts(list): each element's type is str in python2.7
        Returns:
             texts(list): each element's type is unicode in python2.7
        """
        if six.PY2:
            unicode_texts = []
            for text in texts:
                if not isinstance(text, six.string_types):
                    unicode_texts.append(
                        text.decode(sys_stdin_encoding()).decode("utf8"))
                else:
                    unicode_texts.append(text)
            texts = unicode_texts
        return texts

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.predict(
            texts=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU for prediction")

        self.arg_config_group.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help="batch size for prediction")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_file',
            type=str,
            default=None,
            help="file contain input data")
        self.arg_input_group.add_argument(
            '--input_text', type=str, default=None, help="text to predict")

    def check_input_data(self, args):
        input_data = []
        if args.input_file:
            if not os.path.exists(args.input_file):
                print("File %s is not exist." % args.input_file)
                raise RuntimeError
            else:
                input_data = txt_parser.parse(args.input_file, use_strip=True)
        elif args.input_text:
            if args.input_text.strip() != '':
                if six.PY2:
                    input_data = [
                        args.input_text.decode(
                            sys_stdin_encoding()).decode("utf8")
                    ]
                else:
                    input_data = [args.input_text]
            else:
                print(
                    "ERROR: The input data is inconsistent with expectations.")

        if input_data == []:
            print("ERROR: The input data is inconsistent with expectations.")
            raise DataFormatError

        return input_data


class _TransformerEmbeddingTask(hub.BaseTask):
    def __init__(self,
                 pooled_feature,
                 seq_feature,
                 feed_list,
                 data_reader,
                 config=None):
        main_program = pooled_feature.block.program
        super(_TransformerEmbeddingTask, self).__init__(
            main_program=main_program,
            data_reader=data_reader,
            feed_list=feed_list,
            config=config,
            metrics_choices=[])
        self.pooled_feature = pooled_feature
        self.seq_feature = seq_feature

    def _build_net(self):
        return [self.pooled_feature, self.seq_feature]

    def _postprocessing(self, run_states):
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results
            batch_pooled_features = batch_result[0]
            batch_seq_features = batch_result[1]
            for i in range(len(batch_pooled_features)):
                results.append(
                    [batch_pooled_features[i], batch_seq_features[i]])
        return results


class TransformerModule(NLPBaseModule):
    """
    Tranformer Module base class can be used by BERT, ERNIE, RoBERTa and so on.
    """

    def init_pretraining_params(self, exe, pretraining_params_path,
                                main_program):
        assert os.path.exists(
            pretraining_params_path
        ), "[%s] cann't be found." % pretraining_params_path

        def existed_params(var):
            if not isinstance(var, fluid.framework.Parameter):
                return False
            return os.path.exists(
                os.path.join(pretraining_params_path, var.name))

        fluid.io.load_vars(
            exe,
            pretraining_params_path,
            main_program=main_program,
            predicate=existed_params)
        logger.info("Load pretraining parameters from {}.".format(
            pretraining_params_path))

    def context(
            self,
            max_seq_len=128,
            trainable=True,
    ):
        """
        get inputs, outputs and program from pre-trained module

        Args:
            max_seq_len (int): the max sequence length
            trainable (bool): optimizing the pre-trained module params during training or not

        Returns: inputs, outputs, program.
                 The inputs is a dict with keys named input_ids, position_ids, segment_ids, input_mask and task_ids
                 The outputs is a dict with two keys named pooled_output and sequence_output.

        """

        assert max_seq_len <= self.MAX_SEQ_LEN and max_seq_len >= 1, "max_seq_len({}) should be in the range of [1, {}]".format(
            max_seq_len, self.MAX_SEQ_LEN)

        module_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(module_program, startup_program):
            with fluid.unique_name.guard("@HUB_%s@" % self.name):
                input_ids = fluid.layers.data(
                    name='input_ids',
                    shape=[-1, max_seq_len, 1],
                    dtype='int64',
                    lod_level=0)
                position_ids = fluid.layers.data(
                    name='position_ids',
                    shape=[-1, max_seq_len, 1],
                    dtype='int64',
                    lod_level=0)
                segment_ids = fluid.layers.data(
                    name='segment_ids',
                    shape=[-1, max_seq_len, 1],
                    dtype='int64',
                    lod_level=0)
                input_mask = fluid.layers.data(
                    name='input_mask',
                    shape=[-1, max_seq_len, 1],
                    dtype='float32',
                    lod_level=0)
                pooled_output, sequence_output = self.net(
                    input_ids, position_ids, segment_ids, input_mask)

        inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
        }

        outputs = {
            "pooled_output": pooled_output,
            "sequence_output": sequence_output,
            0: pooled_output,
            1: sequence_output
        }

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        # To be compatible with the module v1
        if self.name not in [
                "roberta_wwm_ext_chinese_L-12_H-768_A-12_distillation",
                "roberta_wwm_ext_chinese_L-24_H-1024_A-16_distillation",
                "bert_multi_uncased_L-12_H-768_A-12",
                "bert_cased_L-24_H-1024_A-16"
        ]:
            if self.name == "ernie_tiny":
                prefix = "@HUB_ernie-tiny@"
            elif self.name == "ernie":
                prefix = "@HUB_ernie-stable@"
            elif self.name == "bert_wwm_ext_chinese_L-12_H-768_A-12":
                prefix = "@HUB_bert_wwm_ext_chinese_L-12_H-768_A-12_wwm_ext@"
            else:
                prefix = "@HUB_%s@" % self.name

            vars = filter(lambda var: "tmp" not in var,
                          list(module_program.global_block().vars.keys())[4:])
            paddle_helper.add_vars_prefix(
                program=module_program, prefix=prefix, vars=vars)
        self.init_pretraining_params(
            exe, self.params_path, main_program=module_program)

        self.params_layer = {}
        for param in module_program.global_block().iter_parameters():
            param.trainable = trainable
            match = re.match(r'.*layer_(\d+).*', param.name)
            if match:
                # layer num begins from 0
                layer = match.group(1)
                self.params_layer[param.name] = int(layer)

        return inputs, outputs, module_program

    def get_embedding(self, texts, use_gpu=False, batch_size=1):
        """
        get pooled_output and sequence_output for input texts.
        Warnings: this method depends on Paddle Inference Library, it may not work properly in PaddlePaddle <= 1.6.2.

        Args:
            texts (list): each element is a text sample, each sample include text_a and text_b where text_b can be omitted.
                          for example: [[sample0_text_a, sample0_text_b], [sample1_text_a, sample1_text_b], ...]
            use_gpu (bool): use gpu or not, default False.
            batch_size (int): the data batch size, default 1.

        Returns:
            pooled_outputs(list): its element is a numpy array, the first feature of each text sample.
            sequence_outputs(list): its element is a numpy array, the whole features of each text sample.
        """
        if not hasattr(
                self, "emb_job"
        ) or self.emb_job["batch_size"] != batch_size or self.emb_job[
                "use_gpu"] != use_gpu:
            inputs, outputs, program = self.context(
                trainable=True, max_seq_len=self.MAX_SEQ_LEN)

            reader = hub.reader.ClassifyReader(
                dataset=None,
                vocab_path=self.get_vocab_path(),
                max_seq_len=self.MAX_SEQ_LEN,
                sp_model_path=self.get_spm_path() if hasattr(
                    self, "get_spm_path") else None,
                word_dict_path=self.get_word_dict_path() if hasattr(
                    self, "word_dict_path") else None)

            feed_list = [
                inputs["input_ids"].name,
                inputs["position_ids"].name,
                inputs["segment_ids"].name,
                inputs["input_mask"].name,
            ]

            pooled_feature, seq_feature = outputs["pooled_output"], outputs[
                "sequence_output"]

            config = hub.RunConfig(
                use_data_parallel=False,
                use_cuda=use_gpu,
                batch_size=batch_size)

            self.emb_job = {}
            self.emb_job["task"] = _TransformerEmbeddingTask(
                pooled_feature=pooled_feature,
                seq_feature=seq_feature,
                feed_list=feed_list,
                data_reader=reader,
                config=config,
            )
            self.emb_job["batch_size"] = batch_size
            self.emb_job["use_gpu"] = use_gpu

        return self.emb_job["task"].predict(
            data=texts, return_result=True, accelerate_mode=True)

    def get_spm_path(self):
        if hasattr(self, "spm_path"):
            return self.spm_path
        else:
            return None

    def get_word_dict_path(self):
        if hasattr(self, "word_dict_path"):
            return self.word_dict_path
        else:
            return None

    def get_params_layer(self):
        if not hasattr(self, "params_layer"):
            raise AttributeError(
                "The module context has not been initialized. "
                "Please call context() before using get_params_layer")
        return self.params_layer
