# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import json
import math
import os
import six

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
import paddlehub as hub
from paddlehub.common.paddle_helper import add_vars_prefix, get_variable_info
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import serving
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable

from simnet_bow.processor import load_vocab, preprocess, postprocess


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


@moduleinfo(
    name="simnet_bow",
    version="1.2.0",
    summary=
    "Baidu's open-source similarity network model based on bow_pairwise.",
    author="baidu-nlp",
    author_email="",
    type="nlp/sentiment_analysis")
class SimnetBow(hub.Module):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "infer_model")
        self.vocab_path = os.path.join(self.directory, "assets", "vocab.txt")
        self.vocab = load_vocab(self.vocab_path)
        self.param_file = os.path.join(self.directory, "assets", "params.txt")
        self._word_seg_module = None

        self._set_config()

    @property
    def word_seg_module(self):
        """
        lac module
        """
        if not self._word_seg_module:
            self._word_seg_module = hub.Module(name="lac")
        return self._word_seg_module

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_ir_optim(False)
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

    def context(self, trainable=False, max_seq_len=128, num_slots=1):
        """
        Get the input ,output and program of the pretrained simnet_bow

        Args:
             trainable(bool): whether fine-tune the pretrained parameters of simnet_bow or not。
             max_seq_len (int): It will limit the total sequence returned so that it has a maximum length.
             num_slots(int): It's number of data inputted to the model, selectted as following options:

                 - 1(default): There's only one data to be feeded in the model, e.g. the module is used for sentence classification task.
                 - 2: There are two data to be feeded in the model, e.g. the module is used for text matching task (point-wise).
                 - 3: There are three data to be feeded in the model, e.g. the module is used for text matching task (pair-wise).

        Returns:
             inputs(dict): the input variables of simnet_bow (words)
             outputs(dict): the output variables of input words (word embeddings) and sequence lenght of the first input_text
             main_program(Program): the main_program of simnet_bow with pretrained prameters
        """
        assert num_slots >= 1 and num_slots <= 3, "num_slots must be 1, 2, or 3, but the input is %d" % num_slots
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            text_1 = fluid.layers.data(
                name="text",
                shape=[-1, max_seq_len, 1],
                dtype="int64",
                lod_level=0)
            seq_len = fluid.layers.data(
                name="seq_len", shape=[1], dtype='int64', lod_level=0)
            seq_len_used = fluid.layers.squeeze(seq_len, axes=[1])

            # Add embedding layer.
            w_param_attrs = fluid.ParamAttr(
                name="emb",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                trainable=trainable)
            dict_dim = 500002
            emb_1 = fluid.layers.embedding(
                input=text_1,
                size=[dict_dim, 128],
                is_sparse=True,
                padding_idx=dict_dim - 1,
                dtype='float32',
                param_attr=w_param_attrs)
            emb_1_name = emb_1.name
            data_list = [text_1]
            emb_name_list = [emb_1_name]

            if num_slots > 1:
                text_2 = fluid.data(
                    name='text_2',
                    shape=[-1, max_seq_len],
                    dtype='int64',
                    lod_level=0)
                emb_2 = fluid.embedding(
                    input=text_2,
                    size=[dict_dim, 128],
                    is_sparse=True,
                    padding_idx=dict_dim - 1,
                    dtype='float32',
                    param_attr=w_param_attrs)
                emb_2_name = emb_2.name
                data_list.append(text_2)
                emb_name_list.append(emb_2_name)

            if num_slots > 2:
                text_3 = fluid.data(
                    name='text_3',
                    shape=[-1, max_seq_len],
                    dtype='int64',
                    lod_level=0)
                emb_3 = fluid.embedding(
                    input=text_3,
                    size=[dict_dim, 128],
                    is_sparse=True,
                    padding_idx=dict_dim - 1,
                    dtype='float32',
                    param_attr=w_param_attrs)
                emb_3_name = emb_3.name
                data_list.append(text_3)
                emb_name_list.append(emb_3_name)

            variable_names = filter(
                lambda v: v not in ['text', 'text_2', 'text_3', "seq_len"],
                list(main_program.global_block().vars.keys()))
            prefix_name = "@HUB_{}@".format(self.name)
            add_vars_prefix(
                program=main_program, prefix=prefix_name, vars=variable_names)

            for param in main_program.global_block().iter_parameters():
                param.trainable = trainable

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            # Load the senta_lstm pretrained model.
            def if_exist(var):
                return os.path.exists(
                    os.path.join(self.pretrained_model_path, var.name))

            fluid.io.load_vars(
                exe, self.pretrained_model_path, predicate=if_exist)

            inputs = {'seq_len': seq_len}
            outputs = {}
            for index, data in enumerate(data_list):
                if index == 0:
                    inputs['text'] = data
                    outputs['emb'] = main_program.global_block().vars[
                        prefix_name + emb_name_list[0]]
                else:
                    inputs['text_%s' % (index + 1)] = data
                    outputs['emb_%s' % (index + 1)] = main_program.global_block(
                    ).vars[prefix_name + emb_name_list[index]]
            return inputs, outputs, main_program

    def texts2tensor(self, texts):
        """
        Tranform the texts(dict) to PaddleTensor
        Args:
             texts(list): texts
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
                if isinstance(text, six.string_types):
                    unicode_texts.append(
                        text.decode(sys_stdin_encoding()).decode("utf8"))
                else:
                    unicode_texts.append(text)
            texts = unicode_texts
        return texts

    def check_data(self, texts=[], data={}):
        """
        check input data
        Args:
             texts(list): the input texts to be predicted which the first element is text_1(list)
                          and the second element is text_2(list), such as [['这道题很难'], ['这道题不简单']]
                          if texts not data.
             data(dict): key must be 'text_1' and 'text_2', value is the texts(list) to be predicted
        Returns:
             results(dict): predicted data
        """
        predicted_data = {'text_1': [], 'text_2': []}
        if texts != [] and isinstance(texts, list) and len(texts) == 2 and (len(
                texts[0]) == len(
                    texts[1])) and texts[0] and texts[1] and data == {}:

            predicted_data['text_1'] = texts[0]
            predicted_data['text_2'] = texts[1]

        elif texts == [] and isinstance(data, dict) and isinstance(
                data.get('text_1', None), list) and isinstance(
                    data.get('text_2', None),
                    list) and (len(data['text_1']) == len(
                        data['text_2'])) and data['text_1'] and data['text_2']:

            predicted_data = data

        else:
            raise ValueError(
                "The input data is inconsistent with expectations.")

        return predicted_data

    @serving
    def similarity(self, texts=[], data={}, use_gpu=False, batch_size=1):
        """
        Get the sentiment prediction results results with the texts as input
        Args:
             texts(list): the input texts to be predicted which the first element is text_1(list)
                          and the second element is text_2(list), such as [['这道题很难'], ['这道题不简单']]
                          if texts not data.
             data(dict): key must be 'text_1' and 'text_2', value is the texts(list) to be predicted
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch
        Returns:
             results(list): the word segmentation results
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        data = self.check_data(texts, data)

        start_idx = 0
        iteration = int(math.ceil(len(data['text_1']) / batch_size))
        results = []
        for i in range(iteration):
            batch_data = {'text_1': [], 'text_2': []}
            if i < (iteration - 1):
                batch_data['text_1'] = data['text_1'][start_idx:(
                    start_idx + batch_size)]
                batch_data['text_2'] = data['text_2'][start_idx:(
                    start_idx + batch_size)]
            else:
                batch_data['text_1'] = data['text_1'][start_idx:(
                    start_idx + batch_size)]
                batch_data['text_2'] = data['text_2'][start_idx:(
                    start_idx + batch_size)]
            start_idx = start_idx + batch_size
            processed_results = preprocess(self.word_seg_module, self.vocab,
                                           batch_data, use_gpu, batch_size)

            tensor_words_1 = self.texts2tensor(processed_results["text_1"])
            tensor_words_2 = self.texts2tensor(processed_results["text_2"])

            if use_gpu:
                batch_out = self.gpu_predictor.run(
                    [tensor_words_1, tensor_words_2])
            else:
                batch_out = self.cpu_predictor.run(
                    [tensor_words_1, tensor_words_2])
            batch_result = postprocess(batch_out[1], processed_results)
            results += batch_result
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the simnet_bow module.",
            prog='hub run simnet_bow',
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

        results = self.similarity(
            data=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)

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
            '--text_1', type=str, default=None, help="text to predict")
        self.arg_input_group.add_argument(
            '--text_2', type=str, default=None, help="text to predict")

    def check_input_data(self, args):
        input_data = {}
        if args.input_file:
            if not os.path.exists(args.input_file):
                print("File %s is not exist." % args.input_file)
                raise RuntimeError
            else:
                input_data = txt_parser.parse(args.input_file, use_strip=True)
        elif args.text_1 and args.text_2:
            if args.text_1.strip() != '' and args.text_2.strip() != '':
                if six.PY2:
                    input_data = {
                        "text_1": [
                            args.text_1.strip().decode(
                                sys_stdin_encoding()).decode("utf8")
                        ],
                        "text_2": [
                            args.text_2.strip().decode(
                                sys_stdin_encoding()).decode("utf8")
                        ]
                    }
                else:
                    input_data = {
                        "text_1": [args.text_1],
                        "text_2": [args.text_2]
                    }
            else:
                print(
                    "ERROR: The input data is inconsistent with expectations.")

        if input_data == {}:
            print("ERROR: The input data is inconsistent with expectations.")
            raise DataFormatError

        return input_data

    def get_vocab_path(self):
        """
        Get the path to the vocabulary whih was used to pretrain
        Returns:
             self.vocab_path(str): the path to vocabulary
        """
        return self.vocab_path


if __name__ == "__main__":

    simnet_bow = SimnetBow()
    inputs, outputs, program = simnet_bow.context(num_slots=3)
    print(inputs)
    print(outputs)

    # Data to be predicted
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

    inputs = {"text_1": test_text_1, "text_2": test_text_2}
    results = simnet_bow.similarity(data=inputs, batch_size=2)
    print(results)
    max_score = -1
    result_text = ""
    for result in results:
        if result['similarity'] > max_score:
            max_score = result['similarity']
            result_text = result['text_2']

    print("The most matching with the %s is %s" % (test_text_1[0], result_text))
