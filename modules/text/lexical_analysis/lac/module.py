# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import io
import json
import math
import os

import numpy as np
import paddle
import six
from lac.custom import Customization
from lac.processor import load_kv_dict
from lac.processor import parse_result
from lac.processor import word_to_ids
from paddle.inference import Config
from paddle.inference import create_predictor

import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.common.paddle_helper import add_vars_prefix
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


class DataFormatError(Exception):

    def __init__(self, *args):
        self.args = args


@moduleinfo(
    name="lac",
    version="2.2.1",
    summary=
    "Baidu's open-source lexical analysis tool for Chinese, including word segmentation, part-of-speech tagging & named entity recognition",
    author="baidu-nlp",
    author_email="paddle-dev@baidu.com",
    type="nlp/lexical_analysis")
class LAC(hub.Module):

    def _initialize(self, user_dict=None):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "infer_model")
        self.word2id_dict = load_kv_dict(os.path.join(self.directory, "assets/word.dic"), reverse=True, value_func=int)
        self.id2word_dict = load_kv_dict(os.path.join(self.directory, "assets/word.dic"))
        self.label2id_dict = load_kv_dict(os.path.join(self.directory, "assets/tag.dic"), reverse=True, value_func=int)
        self.id2label_dict = load_kv_dict(os.path.join(self.directory, "assets/tag.dic"))
        self.word_replace_dict = load_kv_dict(os.path.join(self.directory, "assets/q2b.dic"))
        self.oov_id = self.word2id_dict['OOV']
        self.word_dict_len = max(map(int, self.word2id_dict.values())) + 1
        self.label_dict_len = max(map(int, self.label2id_dict.values())) + 1
        self.tag_file = os.path.join(self.directory, "assets/tag_file.txt")

        if user_dict:
            self.set_user_dict(dict_path=user_dict)
        else:
            self.custom = None

        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = Config(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = Config(self.pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_predictor(gpu_config)

    def set_user_dict(self, dict_path, sep=None):
        """
        Set the costomized dictionary if you wanna exploit the self-defined dictionary

        Args:
             dict_path(str): The directory to the costomized dictionary.
             sep: The seperation token in phases. Default as ' ' or '\t'.
        """
        if not os.path.exists(dict_path):
            raise RuntimeError("File %s is not exist." % dict_path)
        self.custom = Customization()
        self.custom.load_customization(dict_path, sep)

    def del_user_dict(self):
        """
        Delete the costomized dictionary if you don't wanna exploit the self-defined dictionary any longer
        """

        if self.custom:
            self.custom = None
            print("Successfully delete the customized dictionary!")

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
                    unicode_texts.append(text.decode(sys_stdin_encoding()).decode("utf8"))
                else:
                    unicode_texts.append(text)
            texts = unicode_texts
        return texts

    def preprocess(self, texts):
        """
        Tranform the texts(list) to PaddleTensor
        Args:
             texts(list): texts
        Returns:
             np.array, list, list
        """
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            text_inds = word_to_ids(text, self.word2id_dict, self.word_replace_dict, oov_id=self.oov_id)
            data += text_inds
            lod.append(len(text_inds) + lod[i])
        return np.array(data).astype('int64'), [lod], [lod[-1], 1]

    def _get_index(self, data_list, item=""):
        """
        find all indexes of item in data_list
        """
        res = []
        for index, data in enumerate(data_list):
            if data == item:
                res.append(index)
        return res

    @serving
    def cut(self, text, use_gpu=False, batch_size=1, return_tag=True):
        """
        The main function that segments an entire text that contains
        Chinese characters into separated words.
        Args:
            text(:obj:`str` or :obj:`List[str]`): The chinese texts to be segmented. This can be a string, a list of strings.
            use_gpu(bool): whether use gpu to predict or not
            batch_size(int): the program deals once with one batch
            return_tag: Whether to get tag or not.

        Returns:
            results(dict or list): The word segmentation result of the input text, whose key is 'word', if text is a list.
                If text is a str, the word segmentation result (list) is obtained.

        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        if isinstance(text, list) and len(text) != 0:

            predicted_data = self.to_unicode(text)

            # drop the empty string like "" in predicted_data
            empty_str_indexes = self._get_index(predicted_data)
            predicted_data = [data for data in predicted_data if data != ""]

            start_idx = 0
            iteration = int(math.ceil(len(predicted_data) / batch_size))
            results = []
            for i in range(iteration):
                if i < (iteration - 1):
                    batch_data = predicted_data[start_idx:(start_idx + batch_size)]
                else:
                    batch_data = predicted_data[start_idx:]

                start_idx = start_idx + batch_size
                data, lod, shape = self.preprocess(batch_data)

                predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
                input_names = predictor.get_input_names()
                input_handle = predictor.get_input_handle(input_names[0])
                input_handle.copy_from_cpu(data)
                input_handle.set_lod(lod)
                input_handle.reshape(shape)

                predictor.run()
                output_names = predictor.get_output_names()
                output_handle = predictor.get_output_handle(output_names[0])

                batch_result = parse_result(batch_data, output_handle, self.id2label_dict, interventer=self.custom)
                results += batch_result

            for index in empty_str_indexes:
                results.insert(index, {"word": [""], "tag": [""]})

            if not return_tag:
                for result in results:
                    result = result.pop("tag")
                return results

            return results
        elif isinstance(text, str) and text != "":
            data, lod, shape = self.preprocess([text])

            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(data)
            input_handle.set_lod(lod)
            input_handle.reshape(shape)

            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])

            batch_result = parse_result([text], output_handle, self.id2label_dict, interventer=self.custom)

            return batch_result[0]['word']
        elif text == "":
            return text
        else:
            raise TypeError("The input data is inconsistent with expectations.")

    def lexical_analysis(self, texts=[], data={}, use_gpu=False, batch_size=1, return_tag=True):
        """
        Get the word segmentation results with the texts as input

        Args:
             texts(list): the input texts to be segmented, if texts not data
             data(dict): key must be 'text', value is the texts to be segmented, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch
             return_tag: Whether to get tag or not.

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

        if texts != [] and isinstance(texts, list) and data == {}:
            predicted_data = texts
        elif texts == [] and isinstance(data, dict) and isinstance(data.get('text', None), list) and data['text']:
            predicted_data = data["text"]
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        predicted_data = self.to_unicode(predicted_data)

        # drop the empty string like "" in predicted_data
        empty_str_indexes = self._get_index(predicted_data)
        predicted_data = [data for data in predicted_data if data != ""]

        start_idx = 0
        iteration = int(math.ceil(len(predicted_data) / batch_size))
        results = []
        for i in range(iteration):
            if i < (iteration - 1):
                batch_data = predicted_data[start_idx:(start_idx + batch_size)]
            else:
                batch_data = predicted_data[start_idx:]

            start_idx = start_idx + batch_size
            data, lod, shape = self.preprocess(batch_data)

            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(data)
            input_handle.set_lod(lod)
            input_handle.reshape(shape)

            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])

            batch_result = parse_result(batch_data, output_handle, self.id2label_dict, interventer=self.custom)
            results += batch_result

        for index in empty_str_indexes:
            results.insert(index, {"word": [""], "tag": [""]})

        if not return_tag:
            for result in results:
                result = result.pop("tag")
            return results

        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(description="Run the lac module.",
                                              prog='hub run lac',
                                              usage='%(prog)s',
                                              add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        if args.user_dict:
            self.set_user_dict(args.user_dict)

        results = self.lexical_analysis(texts=input_data,
                                        use_gpu=args.use_gpu,
                                        batch_size=args.batch_size,
                                        return_tag=args.return_tag)

        return results

    def get_tags(self):
        """
        Get the tags which was used when pretraining lac

        Returns:
             self.tag_name_dict(dict):lac tags
        """
        self.tag_name_dict = {}
        with io.open(self.tag_file, encoding="utf8") as f:
            for line in f:
                tag, tag_name = line.strip().split(" ")
                self.tag_name_dict[tag] = tag_name
        return self.tag_name_dict

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU or not")

        self.arg_config_group.add_argument('--batch_size', type=int, default=1, help="batch size for prediction")
        self.arg_config_group.add_argument('--user_dict',
                                           type=str,
                                           default=None,
                                           help="customized dictionary for intervening the word segmentation result")
        self.arg_config_group.add_argument('--return_tag',
                                           type=ast.literal_eval,
                                           default=True,
                                           help="whether return tags of results or not")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument('--input_file', type=str, default=None, help="file contain input data")
        self.arg_input_group.add_argument('--input_text', type=str, default=None, help="text to predict")

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
                    input_data = [args.input_text.decode(sys_stdin_encoding()).decode("utf8")]
                else:
                    input_data = [args.input_text]

        if input_data == []:
            print("ERROR: The input data is inconsistent with expectations.")
            raise DataFormatError

        return input_data
