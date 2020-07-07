# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
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
#################################################################################
"""
本文件初始化配置和环境的相关类
"""

import ast
import argparse
import configparser
import logging

import os
import math
import pickle

import numpy as np
from paddle import fluid
from paddle.fluid import dygraph

from parser.utils import corpus
from parser.utils import field
from parser.utils import utils
from parser.utils import Embedding


class ArgumentGroup(object):
    """ArgumentGroup"""

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, *args, **kwargs):
        self._group.add_argument(*args, **kwargs)


class ArgConfig(configparser.ConfigParser):
    def __init__(self):
        """定义ArgConfig类，接收参数"""
        super(ArgConfig, self).__init__()

        parser = argparse.ArgumentParser(
            description="BaiDu's Denpendency Parser.")
        model_g = ArgumentGroup(parser, "model",
                                "model configuration and paths.")
        model_g.add_arg(
            '--mode',
            default='train',
            choices=['train', 'evaluate', 'predict', 'predict_q'],
            help='choices of additional features')
        model_g.add_arg(
            '--config_path',
            '-c',
            default='config.ini',
            help='path to config file')
        model_g.add_arg(
            '--output_dir',
            default='exp/baidu',
            help='Directory path to save model and field.')

        data_g = ArgumentGroup(
            parser, "data",
            "Data paths, vocab paths and data processing options")
        data_g.add_arg('--train_data_path', help='path to training data.')
        data_g.add_arg('--valid_data_path', help='path to valid data.')
        data_g.add_arg('--test_data_path', help='path to testing data.')
        data_g.add_arg('--infer_data_path', help='path to dataset')
        data_g.add_arg(
            '--pretrained_embedding_dir',
            "--pre_emb",
            help='path to pretrained embeddings')
        data_g.add_arg(
            '--batch_size', default=16000, type=int, help='batch size')

        log_g = ArgumentGroup(parser, "logging", "logging related")
        log_g.add_arg('--log_path', default='./log/log', help='log path')
        log_g.add_arg(
            '--infer_result_path',
            default='infer_result',
            help="Directory path to infer result.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg(
            '--use_cuda',
            '-gpu',
            action='store_true',
            help='If set, use GPU for training.')
        run_type_g.add_arg(
            '--preprocess',
            '-p',
            action='store_true',
            help='whether to preprocess the data first')
        run_type_g.add_arg(
            '--use_data_parallel',
            action='store_true',
            help=
            'The flag indicating whether to use data parallel mode to train the model.'
        )
        run_type_g.add_arg(
            '--seed',
            '-s',
            default=1,
            type=int,
            help='seed for generating random numbers')
        run_type_g.add_arg(
            '--threads', '-t', default=16, type=int, help='max num of threads')
        run_type_g.add_arg(
            '--tree',
            action='store_true',
            help='whether to ensure well-formedness')
        run_type_g.add_arg(
            '--prob', action='store_true', help='whether to output probs')

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg(
            '--feat',
            default='char',
            choices=['pos', 'char'],
            help='choices of additional features')
        train_g.add_arg(
            '--buckets', default=32, type=int, help='max num of buckets to use')
        train_g.add_arg(
            '--punct',
            action='store_true',
            help='whether to include punctuation')
        train_g.add_arg(
            '--unk', default='unk', help='unk token in pretrained embeddings')

        custom_g = ArgumentGroup(parser, "customize", "customized options.")
        self.build_conf(parser)

    def build_conf(self, parser):
        """初始化参数，将parser解析的参数和config文件中读取的参数合并"""
        args = parser.parse_args()
        self.read(args.config_path)
        self.namespace = argparse.Namespace()
        self.update(
            dict((name, ast.literal_eval(value)) for section in self.sections()
                 for name, value in self.items(section)))
        args.nranks = fluid.dygraph.ParallelEnv().nranks
        args.local_rank = fluid.dygraph.ParallelEnv().local_rank
        args.fields_path = os.path.join(args.output_dir, 'fields')
        args.model_path = os.path.join(args.output_dir, 'model')
        # update config from args
        self.update(vars(args))
        return self

    def __repr__(self):
        """repr"""
        s = line = "-" * 25 + "-+-" + "-" * 25 + "\n"
        s += f"{'Param':25} | {'Value':^25}\n" + line
        for name, value in vars(self.namespace).items():
            s += f"{name:25} | {str(value):^25}\n"
        s += line

        return s

    def __getattr__(self, attr):
        """getattr"""
        return getattr(self.namespace, attr)

    def __setitem__(self, name, value):
        """setitem"""
        setattr(self.namespace, name, value)

    def __getstate__(self):
        """getstate"""
        return vars(self)

    def __setstate__(self, state):
        """setstate"""
        self.__dict__.update(state)

    def update(self, kwargs):
        """更新参数"""
        for name, value in kwargs.items():
            setattr(self.namespace, name, value)

        return self


class Environment(object):
    """定义Environment类，用于初始化运行环境"""

    def __init__(self, fields_path):
        # init seed
        fluid.default_main_program().random_seed = 1
        np.random.seed(1)

        os.environ['FLAGS_paddle_num_threads'] = str(16)

        logging.info("loading the fields.")
        with open(fields_path, "rb") as f:
            self.fields = pickle.load(f)
        self.WORD, self.FEAT = self.fields.FORM
        self.ARC, self.REL = self.fields.HEAD, self.fields.DEPREL
        self.puncts = np.array(
            [i for s, i in self.WORD.vocab.stoi.items() if utils.ispunct(s)])
