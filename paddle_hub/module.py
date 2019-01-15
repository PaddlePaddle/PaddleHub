#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import numpy as np
import tempfile
import os
import pickle

from collections import defaultdict
from paddle_hub.downloader import download_and_uncompress
from paddle_hub import module_desc_pb2
from paddle_hub.signature import Signature
from paddle_hub.utils import to_list

__all__ = ["Module", "ModuleConfig", "ModuleUtils"]

# paddle hub module dir name
ASSETS_DIRNAME = "assets"
META_DIRNAME = "meta"
MODEL_DIRNAME = "model"
# paddle hub module serialze file name
DICT_FILENAME = "vocab.txt"
PARAM_FILENAME = "param.pkl"
MODULE_DESC_PBNAME = "module_desc.pb"
GENERATOR_FILENAME = "unique_name_generator.pkl"


def mkdir(path):
    """ the same as the shell command mkdir -p "
    """
    if not os.path.exists(path):
        os.makedirs(path)


class Module(object):
    """
    Core object of PaddleHub
    """

    def __init__(self, module_url=None, module_dir=None):
        if module_url == None and module_dir == None:
            raise Exception("Module:module_url and module_dir are None!")

        self.module_dir = ""
        self.module_name = ""
        # donwload module
        if module_url is not None and module_url.startswith("http"):
            # if it's remote url link, then download and uncompress it
            self.module_name, self.module_dir = download_and_uncompress(
                module_url)
            #TODO(ZeyuChen): check url link is valid url
        elif module_dir is not None:
            # otherwise it's local path, no need to deal with it
            self.module_dir = module_dir
            # use the path name as module name by default
            self.module_name = module_dir.split("/")[-1]
            #TODO(ZeyuChen) add more check about loading module from local path

    def _process_parameter(self):
        global_block = self.inference_program.global_block()
        param_path = ModuleConfig.meta_param_path(self.module_dir)
        with open(param_path, "rb") as file:
            param_arr = pickle.load(file)
        for param in param_arr:
            if (param['name'] not in global_block.vars):
                continue
            var = global_block.var(param['name'])
            global_block.create_parameter(
                **param,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                error_clip=var.error_clip,
                stop_gradient=var.stop_gradient,
                is_data=var.is_data)

    def __call__(self, sign_name="default", trainable=False):
        """ Call default signature and return results
        """

        def _set_param_trainable(program, trainable=False):
            for param in program.global_block().iter_parameters():
                param.trainable = trainable

        def _process_op_attr(program, is_test=False):
            for op in program.global_block().ops:
                if op.has_attr("is_test"):
                    op._set_attr("is_test", is_test)

        # load paddle inference model
        place = fluid.CPUPlace()
        model_dir = os.path.join(self.module_dir, MODEL_DIRNAME)
        self.exe = fluid.Executor(fluid.CPUPlace())
        self.inference_program, self.feed_target_names, self.fetch_targets = fluid.io.load_inference_model(
            dirname=os.path.join(model_dir, sign_name, executor=self.exe))

        # remove feed fetch operator and variable
        ModuleUtils.remove_feed_fetch_op(self.inference_program)
        # print("inference_program")
        # print(self.inference_program)
        print("**feed_target_names**\n{}".format(self.feed_target_names))
        print("**fetch_targets**\n{}".format(self.fetch_targets))

        self.config = ModuleConfig(self.module_dir)
        self.config.load()
        self._process_parameter()
        name_generator_path = ModuleConfig.name_generator_path(self.module_dir)
        with open(name_generator_path, "rb") as data:
            generator = pickle.load(data)

        program = self.get_inference_program().clone()

        _process_op_attr(program=program, is_test=False)
        _set_param_trainable(program=program, trainable=trainable)

        return self.feed_target_names, self.fetch_targets, program, generator

    def get_inference_program(self):
        return self.inference_program

    # for text sequence input, transform to lod tensor as paddle graph's input
    def _preprocess_input(self, inputs):
        # words id mapping and dealing with oov
        # transform to lod tensor
        seq = []
        for s in inputs:
            seq.append(self._word_id_mapping(s))

        lod_tensor = self.seq2lod_tensor(seq)

        return lod_tensor

    def seq2lod_tensor(self, seq_inputs, place=fluid.CPUPlace()):
        """ sequence to lod tensor, need to determine which space"""
        lod = []
        lod.append([])
        for s in seq_inputs:
            # generate lod
            lod[0].append(len(s))

        # print("seq", seq_inputs)
        # print("lod", lod)

        lod_tensor = fluid.create_lod_tensor(seq_inputs, lod, place)

        return lod_tensor

    def _word_id_mapping(self, inputs):
        word_dict = self.config.get_dict()
        return list(map(lambda x: word_dict[x], inputs))


class ModuleConfig(object):
    def __init__(self, module_dir, module_name=None):
        # generate model desc protobuf
        self.module_dir = module_dir
        self.desc = module_desc_pb2.ModuleDesc()
        if module_name == None:
            module_name = module_dir.split("/")[-1]
        # initialize module config default value
        self.desc.name = module_name
        self.desc.contain_assets = True
        self.desc.return_numpy = False

        # init dict
        self.dict = defaultdict(int)
        self.dict.setdefault(0)

    def get_dict(self):
        """ Return dictionary in Module"""
        return self.dict

    def load(self):
        """
        Load module config from module directory.
        """
        #TODO(ZeyuChen): check module_desc.pb exsitance
        with open(ModuleConfig.module_desc_path(self.module_dir), "rb") as fi:
            self.desc.ParseFromString(fi.read())

        if self.desc.contain_assets:
            # load assets
            word_id = 0
            with open(ModuleConfig.assets_dict_path(self.module_dir)) as fi:
                words = fi.readlines()
                #TODO(ZeyuChen) check whether word id is duplicated and valid
                for line in fi:
                    w, w_id = line.split()
                    self.dict[w] = int(w_id)

    def return_numpy(self):
        """Return numpy or not according to the proto config.
        """
        return self.desc.return_numpy

    def save_dict(self, word_dict, dict_name=DICT_FILENAME):
        """ Save dictionary for NLP module
        """
        for w in word_dict:
            self.dict[w] = word_dict[w]

    @staticmethod
    def module_desc_path(module_dir):
        return os.path.join(module_dir, MODULE_DESC_PBNAME)

    @staticmethod
    def name_generator_path(module_dir):
        meta_path = os.path.join(module_dir, META_DIRNAME)
        mkdir(meta_path)
        return os.path.join(meta_path, GENERATOR_FILENAME)

    @staticmethod
    def assets_dict_path(module_dir):
        assets_path = os.path.join(module_dir, ASSETS_DIRNAME)
        mkdir(assets_path)
        return os.path.join(assets_path, DICT_FILENAME)

    @staticmethod
    def meta_param_path(module_dir):
        meta_path = os.path.join(module_dir, META_DIRNAME)
        mkdir(meta_path)
        return os.path.join(meta_path, PARAM_FILENAME)

    @staticmethod
    def meta_name_generator_path(module_dir):
        meta_path = os.path.join(module_dir, META_DIRNAME)
        mkdir(meta_path)
        return os.path.join(meta_path, GENERATOR_FILENAME)


def create_module(sign_arr, program, module_dir=None, word_dict=None):
    """ Create a module from main program
    """
    assert isinstance(
        program, fluid.Program), "program should be instance of fluid.Program"
    assert sign_arr, "signature array should not be None"

    if module_dir is None:
        module_dir = os.path.join(".", "hub_module")
    # create module path for saving
    mkdir(module_dir)

    module = module_desc_pb2.ModuleDesc()
    program = program.clone()

    if word_dict is None:
        module.contain_assets = False
    else:
        module.contain_assets = True
        with open(ModuleConfig.assets_dict_path(module_dir), "w") as fo:
            for w in word_dict:
                w_id = word_dict[w]
                fo.write("{}\t{}\n".format(w, w_id))

    # save the unique name generator object
    var_name_arr = [
        '_'.join(var.split('@')[0].split('.')[0].split('_')[0:-1])
        for block in program.blocks for var in block.vars
    ]
    with fluid.unique_name.guard():
        for var_name in var_name_arr:
            fluid.unique_name.generate(var_name)
        generator = fluid.unique_name.generator

    with open(ModuleConfig.name_generator_path(module_dir), "wb") as fo:
        pickle.dump(generator, fo)

    # save fluid Parameter
    param_arr = []
    for param in program.global_block().iter_parameters():
        param_info = {
            'name': param.name,
            'regularizer': param.regularizer,
            'gradient_clip_attr': param.gradient_clip_attr,
            'trainable': param.trainable,
            'optimize_attr': param.optimize_attr,
            'do_model_average': param.do_model_average
        }
        param_arr.append(param_info)

    with open(ModuleConfig.meta_param_path(module_dir), "wb") as fo:
        pickle.dump(param_arr, fo)

    # save signarture info
    sign_map = module.sign2var
    sign_arr = to_list(sign_arr)
    for sign in sign_arr:
        assert isinstance(sign,
                          Signature), "sign_arr should be list of Signature"

        if sign.get_name() in sign_map:
            raise "Error! sign_arr contains repeat signatrue %s" % sign

        var = sign_map[sign.get_name()]
        feed_desc = var.feed_desc
        fetch_desc = var.fetch_desc
        for input in sign.get_inputs():
            feed_var = feed_desc.add()
            feed_var.var_name = input.name

        for output in sign.get_outputs():
            fetch_var = fetch_desc.add()
            fetch_var.var_name = output.name

    # save inference program
    exe = fluid.Executor(place=fluid.CPUPlace())
    model_dir = os.path.join(module_dir, "model")
    mkdir(model_dir)
    # TODO(wuzewu): save paddle model with a more effective way
    for sign in sign_arr:
        save_model_dir = os.path.join(model_dir, sign.get_name())
        fluid.io.save_inference_model(
            save_model_dir,
            feeded_var_names=[var.name for var in sign.get_inputs()],
            target_vars=sign.get_outputs(),
            main_program=program,
            executor=exe)

    # save to disk
    data = module.SerializeToString()
    with open(ModuleConfig.module_desc_path(module_dir), "wb") as f:
        f.write(data)


class ModuleUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def remove_feed_fetch_op(program):
        """ remove feed and fetch operator and variable for fine-tuning
        """
        print("remove feed fetch op")
        block = program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(block.ops):
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            block._remove_op(index)

        block._remove_var("feed")
        block._remove_var("fetch")

        program.desc.flush()

    @staticmethod
    def module_desc_path(module_dir):
        pass
