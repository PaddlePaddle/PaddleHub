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

import ast
import argparse
import configparser
import logging

import os
import pickle

import numpy as np
from paddle import fluid

from DuDepParser.parser.data_struct import utils
from DuDepParser.parser.data_struct import CoNLL
from DuDepParser.parser.data_struct import Corpus
from DuDepParser.parser.data_struct import Embedding
from DuDepParser.parser.data_struct import Field
from DuDepParser.parser.data_struct import SubwordField


class ArgumentGroup(object):
    """ArgumentGroup"""

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, *args, **kwargs):
        self._group.add_argument(*args, **kwargs)


class ArgConfig(configparser.ConfigParser):
    def __init__(self, args=None):
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
            help='Select task mode')
        model_g.add_arg(
            '--config_path',
            '-c',
            default='config.ini',
            help='path to config file')
        model_g.add_arg(
            '--model_files',
            default='model_files/baidu',
            help='Directory path to save model and ')

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
            '--batch_size', default=5000, type=int, help='batch size')

        log_g = ArgumentGroup(parser, "logging", "logging related")
        log_g.add_arg('--log_path', default='./log/log', help='log path')
        log_g.add_arg(
            '--log_level',
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'],
            help='log level')
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
            '--buckets', default=15, type=int, help='max num of buckets to use')
        train_g.add_arg(
            '--punct',
            action='store_true',
            help='whether to include punctuation')
        train_g.add_arg(
            '--unk', default='unk', help='unk token in pretrained embeddings')

        custom_g = ArgumentGroup(parser, "customize", "customized options.")
        self.build_conf(parser, args)

    def build_conf(self, parser, args=None):
        """初始化参数，将parser解析的参数和config文件中读取的参数合并"""
        args = parser.parse_args(args)
        self.read(args.config_path)
        self.namespace = argparse.Namespace()
        self.update(
            dict((name, ast.literal_eval(value)) for section in self.sections()
                 for name, value in self.items(section)))
        args.nranks = fluid.dygraph.ParallelEnv().nranks
        args.local_rank = fluid.dygraph.ParallelEnv().local_rank
        args.fields_path = os.path.join(args.model_files, 'fields')
        args.model_path = os.path.join(args.model_files, 'model')
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

    def __init__(self, args):
        self.args = args
        # init log
        if self.args.log_path:
            utils.init_log(self.args.log_path, self.args.local_rank,
                           self.args.log_level)
        # init seed
        fluid.default_main_program().random_seed = self.args.seed
        np.random.seed(self.args.seed)
        # init place
        if self.args.use_cuda:
            if self.args.use_data_parallel:
                self.place = fluid.CUDAPlace(
                    fluid.dygraph.parallel.Env().dev_id)
            else:
                self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()

        os.environ['FLAGS_paddle_num_threads'] = str(self.args.threads)
        os.makedirs(self.args.model_files, exist_ok=True)

        if not os.path.exists(self.args.fields_path) or self.args.preprocess:
            logging.info("Preprocess the data")
            self.WORD = Field(
                'word', pad=utils.pad, unk=utils.unk, bos=utils.bos, lower=True)
            if self.args.feat == 'char':
                self.FEAT = SubwordField(
                    'chars',
                    pad=utils.pad,
                    unk=utils.unk,
                    bos=utils.bos,
                    fix_len=self.args.fix_len,
                    tokenize=list)
            else:
                self.FEAT = Field('postag', bos=utils.bos)
            self.ARC = Field(
                'head', bos=utils.bos, use_vocab=False, fn=utils.numericalize)
            self.REL = Field('deprel', bos=utils.bos)
            if self.args.feat == 'char':
                self.fields = CoNLL(
                    FORM=(self.WORD, self.FEAT), HEAD=self.ARC, DEPREL=self.REL)
            else:
                self.fields = CoNLL(
                    FORM=self.WORD,
                    CPOS=self.FEAT,
                    HEAD=self.ARC,
                    DEPREL=self.REL)

            train = Corpus.load(self.args.train_data_path, self.fields)
            if self.args.pretrained_embedding_dir:
                logging.info("loading pretrained embedding from file.")
                embed = Embedding.load(self.args.pretrained_embedding_dir,
                                       self.args.unk)
            else:
                embed = None
            self.WORD.build(train, self.args.min_freq, embed)
            self.FEAT.build(train)
            self.REL.build(train)
            if self.args.local_rank == 0:
                with open(self.args.fields_path, "wb") as f:
                    logging.info("dumping fileds to disk.")
                    pickle.dump(self.fields, f, protocol=2)
        else:
            logging.info("loading the fields.")
            with open(self.args.fields_path, "rb") as f:
                self.fields = pickle.load(f)

            if isinstance(self.fields.FORM, tuple):
                self.WORD, self.FEAT = self.fields.FORM
            else:
                self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.ARC, self.REL = self.fields.HEAD, self.fields.DEPREL
        self.puncts = np.array(
            [i for s, i in self.WORD.vocab.stoi.items() if utils.ispunct(s)])

        if self.WORD.embed is not None:
            self.args["pretrained_embed_shape"] = self.WORD.embed.shape
        else:
            self.args["pretrained_embed_shape"] = None

        self.args.update({
            'n_words': self.WORD.vocab.n_init,
            'n_feats': len(self.FEAT.vocab),
            'n_rels': len(self.REL.vocab),
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index,
            'bos_index': self.WORD.bos_index,
            'feat_pad_index': self.FEAT.pad_index
        })
