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
import six

import numpy as np
import paddle.fluid as fluid

from paddle.inference import Config
from paddle.inference import create_predictor

import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.common.paddle_helper import add_vars_prefix
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import moduleinfo, runnable, serving

from lac.network import lex_net
from lac.processor import load_kv_dict, word_to_ids, parse_result
from lac.custom import Customization


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


@moduleinfo(
    name="lac",
    version="2.2.0",
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

    def _get_device_id(self, places):
        try:
            places = os.environ[places]
            id = int(places)
        except:
            id = -1
        return id

    def _set_config(self):
        """
        predictor config setting
        """

        # create default cpu predictor
        cpu_config = Config(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        # create predictors using various types of devices

        # npu
        npu_id = self._get_device_id("FLAGS_selected_npus")
        if npu_id != -1:
            # use npu
            npu_config = Config(self.pretrained_model_path)
            npu_config.disable_glog_info()
            npu_config.enable_npu(device_id=npu_id)
            self.npu_predictor = create_predictor(npu_config)

        # gpu
        gpu_id = self._get_device_id("CUDA_VISIBLE_DEVICES")
        if gpu_id != -1:
            # use gpu
            gpu_config = Config(self.pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=gpu_id)
            self.gpu_predictor = create_predictor(gpu_config)

        # xpu
        xpu_id = self._get_device_id("XPU_VISIBLE_DEVICES")
        if xpu_id != -1:
            # use xpu
            xpu_config = Config(self.pretrained_model_path)
            xpu_config.disable_glog_info()
            xpu_config.enable_xpu(100)
            self.xpu_predictor = create_predictor(xpu_config)

    def _internal_predict(self, predictor, texts):
        """
        Tranform the texts(list) to Tensor and then do "real predict"
        Args:
             texts(list): texts
        Returns:
             result(PaddleInferTensor): predict output
        """

        # texts to data and lod
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            text_inds = word_to_ids(text, self.word2id_dict, self.word_replace_dict, oov_id=self.oov_id)
            data += text_inds
            lod.append(len(text_inds) + lod[i])

        # get predictor tensor
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        # set data, shape and lod
        input_tensor.copy_from_cpu(np.array(data).astype('int64'))
        input_tensor.reshape([lod[-1], 1])
        input_tensor.set_lod([lod])

        # real predict
        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])

        return output_handle

    def context(self, trainable=False):
        """
        Get the input ,output and program of the pretrained lac

        Args:
             trainable(bool): whether fine-tune the pretrained parameters of lac or not

        Returns:
             inputs(dict): the input variables of lac (words)
             outputs(dict): the output variables of lac (the word segmentation results)
             main_program(Program): the main_program of lac with pretrained prameters
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                crf_decode, word, fc = lex_net(self.word_dict_len, self.label_dict_len)
                word_name = word.name
                pred_name = crf_decode.name
                fc_name = fc.name

                prefix_name = "@HUB_{}@".format(self.name)
                add_vars_prefix(program=main_program, prefix=prefix_name)
                for param in main_program.global_block().iter_parameters():
                    param.trainable = trainable

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)

                # load the lac pretrained model
                def if_exist(var):
                    return os.path.exists(os.path.join(self.pretrained_model_path, var.name))

                fluid.io.load_vars(exe, self.pretrained_model_path, predicate=if_exist)

                inputs = {"words": main_program.global_block().vars[prefix_name + word_name]}
                outputs = {
                    "predicted": main_program.global_block().vars[prefix_name + pred_name],
                    "sentence_feature": main_program.global_block().vars[prefix_name + fc_name]
                }
                return inputs, outputs, main_program

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
    def cut(self, text, use_gpu=False, batch_size=1, return_tag=True, use_device=None):
        """
        The main function that segments an entire text that contains
        Chinese characters into separated words.
        Args:
            text(:obj:`str` or :obj:`List[str]`): The chinese texts to be segmented. This can be a string, a list of strings.
            use_gpu(bool): whether use gpu to predict or not
            batch_size(int): the program deals once with one batch
            return_tag: Whether to get tag or not.
            use_device (str): use cpu, gpu, xpu or npu, overwrites use_gpu flag.

        Returns:
            results(dict or list): The word segmentation result of the input text, whose key is 'word', if text is a list.
                If text is a str, the word segmentation result (list) is obtained.

        """

        # real predictor to use
        if use_device is not None:
            if use_device == "cpu":
                predictor = self.cpu_predictor
            elif use_device == "xpu":
                predictor = self.xpu_predictor
            elif use_device == "npu":
                predictor = self.npu_predictor
            elif use_device == "gpu":
                predictor = self.gpu_predictor
            else:
                raise Exception("Unsupported device: " + use_device)
        else:
            # use_device is not set, therefore follow use_gpu
            if use_gpu:
                predictor = self.gpu_predictor
            else:
                predictor = self.cpu_predictor

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
                batch_out = self._internal_predict(predictor, batch_data)
                batch_result = parse_result(batch_data, batch_out, self.id2label_dict, interventer=self.custom)
                results += batch_result

            for index in empty_str_indexes:
                results.insert(index, {"word": [""], "tag": [""]})

            if not return_tag:
                for result in results:
                    result = result.pop("tag")
                return results

            return results
        elif isinstance(text, str) and text != "":
            batch_out = self._internal_predict(predictor, [text])
            batch_result = parse_result([text], batch_out, self.id2label_dict, interventer=self.custom)

            return batch_result[0]['word']
        elif text == "":
            return text
        else:
            raise TypeError("The input data is inconsistent with expectations.")

    def lexical_analysis(self, texts=[], data={}, use_gpu=False, batch_size=1, return_tag=True, use_device=None):
        """
        Get the word segmentation results with the texts as input

        Args:
             texts(list): the input texts to be segmented, if texts not data
             data(dict): key must be 'text', value is the texts to be segmented, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch
             return_tag: Whether to get tag or not.
             use_device (str): use cpu, gpu, xpu or npu, overwrites use_gpu flag.

        Returns:
             results(list): the word segmentation results
        """

        # real predictor to use
        if use_device is not None:
            if use_device == "cpu":
                predictor = self.cpu_predictor
            elif use_device == "xpu":
                predictor = self.xpu_predictor
            elif use_device == "npu":
                predictor = self.npu_predictor
            elif use_device == "gpu":
                predictor = self.gpu_predictor
            else:
                raise Exception("Unsupported device: " + use_device)
        else:
            # use_device is not set, therefore follow use_gpu
            if use_gpu:
                predictor = self.gpu_predictor
            else:
                predictor = self.cpu_predictor

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
            batch_out = self._internal_predict(predictor, batch_data)
            batch_result = parse_result(batch_data, batch_out, self.id2label_dict, interventer=self.custom)
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
                                        return_tag=args.return_tag,
                                        use_device=args.use_device)

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
        self.arg_config_group.add_argument('--use_device',
                                           choices=["cpu", "gpu", "xpu", "npu"],
                                           help="use cpu, gpu, xpu or npu. overwrites use_gpu flag.")

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


if __name__ == '__main__':
    lac = LAC(user_dict="user.dict")
    # or use the fuction user_dict to set
    # lac.set_user_dict("user.dict")

    test_text = ["今天是个好日子", "天气预报说今天要下雨", "", "下一班地铁马上就要到了", "", "调料份量不能多，也不能少，味道才能正好", "", "", "春天的花开秋天的风以及冬天的落阳"]

    # execute predict and print the result
    results = lac.cut(text=test_text, use_gpu=True, batch_size=7, return_tag=True)
    for result in results:
        if six.PY2:
            print(json.dumps(result['word'], encoding="utf8", ensure_ascii=False))
            print(json.dumps(result['tag'], encoding="utf8", ensure_ascii=False))
        else:
            print(result['word'])
            print(result['tag'])

    # delete the costomized dictionary
    lac.del_user_dict()

    results = lac.cut(text="春天的花开秋天的风以及冬天的落阳", use_gpu=False, batch_size=1, return_tag=False)
    print(results)

    # get the tags that was exploited as pretraining lac
    print(lac.get_tags())
