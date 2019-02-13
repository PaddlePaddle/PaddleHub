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

import paddle
import paddle.fluid as fluid
import numpy as np
import tempfile
import os
import copy

from collections import defaultdict
from paddle_hub.downloader import download_and_uncompress
from paddle_hub import module_desc_pb2
from paddle_hub.logger import logger
from paddle_hub.signature import Signature
from paddle_hub.utils import to_list, get_variable_info, mkdir
from paddle_hub.version import __version__

__all__ = ["Module", "ModuleConfig", "ModuleUtils"]

# paddle hub module dir name
ASSETS_DIRNAME = "assets"
META_DIRNAME = "meta"
MODEL_DIRNAME = "model"
# paddle hub module serialze file name
DICT_FILENAME = "vocab.txt"
PARAM_FILENAME = "param.pkl"
MODULE_DESC_PBNAME = "module_desc.pb"
# paddle hub var prefix
HUB_VAR_PREFIX = "@HUB@"


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
        param_attrs = self.config.desc.param_attrs
        for key, param_attr in param_attrs.items():
            param = {}
            param['name'] = HUB_VAR_PREFIX + key
            param['trainable'] = param_attr.trainable
            param['do_model_average'] = param_attr.do_model_average
            param['optimize_attr'] = {}
            param['optimize_attr'][
                'learning_rate'] = param_attr.optimize_attr.m['learning_rate'].f

            # TODO(wuzewu): recover the param attr with a more reliable way
            if param_attr.regularizer.type == "L2DecayRegularizer":
                regularizer = fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=param_attr.regularizer.
                    regularization_coeff)
            elif param_attr.regularizer.type == "L1DecayRegularizer":
                regularizer = fluid.regularizer.L1DecayRegularizer(
                    regularization_coeff=param_attr.regularizer.
                    regularization_coeff)
            else:
                regularizer = None
            param['regularizer'] = regularizer

            if param_attr.gradient_clip_attr.type == "ErrorClipByValue":
                clip = fluid.clip.ErrorClipByValue(
                    max=param_attr.gradient_clip_attr.max,
                    min=param_attr.gradient_clip_attr.min)
            elif param_attr.gradient_clip_attr.type == "GradientClipByValue":
                clip = fluid.clip.GradientClipByValue(
                    max=param_attr.gradient_clip_attr.max,
                    min=param_attr.gradient_clip_attr.min)
            elif param_attr.gradient_clip_attr.type == "GradientClipByNorm":
                clip = fluid.clip.GradientClipByNorm(
                    clip_norm=param_attr.gradient_clip_attr.clip_norm)
            elif param_attr.gradient_clip_attr.type == "GradientClipByGlobalNorm":
                clip = fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=param_attr.gradient_clip_attr.clip_norm,
                    group_name=param_attr.gradient_clip_attr.group_name)
            else:
                clip = None
            param['gradient_clip_attr'] = clip

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

        def _process_input_output_key(module_desc, signature):
            signature = module_desc.sign2var[signature]

            feed_dict = {}
            fetch_dict = {}

            for index, feed in enumerate(signature.feed_desc):
                if feed.alias != "":
                    feed_dict[feed.alias] = feed.var_name
                feed_dict[index] = feed.var_name

            for index, fetch in enumerate(signature.fetch_desc):
                if fetch.alias != "":
                    fetch_dict[fetch.alias] = fetch.var_name
                fetch_dict[index] = fetch.var_name

            return feed_dict, fetch_dict

        self.config = ModuleConfig(self.module_dir)
        self.config.load()
        # load paddle inference model
        place = fluid.CPUPlace()
        model_dir = os.path.join(self.module_dir, MODEL_DIRNAME)
        self.exe = fluid.Executor(fluid.CPUPlace())
        self.inference_program, self.feed_target_names, self.fetch_targets = fluid.io.load_inference_model(
            dirname=os.path.join(model_dir, sign_name), executor=self.exe)

        feed_dict, fetch_dict = _process_input_output_key(
            self.config.desc, sign_name)

        # remove feed fetch operator and variable
        ModuleUtils.remove_feed_fetch_op(self.inference_program)
        logger.info("**feed_target_names**\n{}".format(self.feed_target_names))
        logger.info("**fetch_targets**\n{}".format(self.fetch_targets))
        self._process_parameter()

        program = self.get_inference_program().clone()

        _process_op_attr(program=program, is_test=False)
        _set_param_trainable(program=program, trainable=trainable)

        for key, value in feed_dict.items():
            var = program.global_block().var(HUB_VAR_PREFIX + value)
            feed_dict[key] = var

        for key, value in fetch_dict.items():
            var = program.global_block().var(HUB_VAR_PREFIX + value)
            fetch_dict[key] = var

        return feed_dict, fetch_dict, program

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
        word_dict = self.config.get_assets_vocab()
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

    def get_assets_vocab(self):
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
    def assets_dict_path(module_dir):
        assets_path = os.path.join(module_dir, ASSETS_DIRNAME)
        mkdir(assets_path)
        return os.path.join(assets_path, DICT_FILENAME)

    @staticmethod
    def meta_param_path(module_dir):
        meta_path = os.path.join(module_dir, META_DIRNAME)
        mkdir(meta_path)
        return os.path.join(meta_path, PARAM_FILENAME)


def create_module(sign_arr, module_dir=None, word_dict=None):
    """ Create a module from main program
    """
    assert sign_arr, "signature array should not be None"

    # check all variable
    sign_arr = to_list(sign_arr)
    program = sign_arr[0].get_inputs()[0].block.program
    for sign in sign_arr:
        assert isinstance(sign,
                          Signature), "sign_arr should be list of Signature"

        for input in sign.get_inputs():
            _tmp_program = input.block.program
            assert program == _tmp_program, "all the variable should come from the same program"

        for output in sign.get_outputs():
            _tmp_program = output.block.program
            assert program == _tmp_program, "all the variable should come from the same program"

    # create module path for saving
    if module_dir is None:
        module_dir = os.path.join(".", "hub_module")
    mkdir(module_dir)

    # create module pb
    module_desc = module_desc_pb2.ModuleDesc()
    module_desc.auth_info.hub_version = __version__
    module_desc.auth_info.paddle_version = paddle.__version__
    logger.info("hub version is %s" % __version__)
    logger.info("paddle version is %s" % paddle.__version__)
    program = program.clone()

    # save asset
    if word_dict is None:
        module_desc.contain_assets = False
    else:
        module_desc.contain_assets = True
        with open(ModuleConfig.assets_dict_path(module_dir), "w") as fo:
            for w in word_dict:
                w_id = word_dict[w]
                fo.write("{}\t{}\n".format(w, w_id))

    # save fluid Parameter
    param_attrs = module_desc.param_attrs
    for param in program.global_block().iter_parameters():
        param_attr = param_attrs[param.name]
        param_attr.trainable = param.trainable
        if param.do_model_average:
            param_attr.do_model_average = param.do_model_average
        # TODO(wuzewu): add a func to transfer python dict to fexiable data
        param_attr.optimize_attr.type = module_desc_pb2.MAP
        param_attr.optimize_attr.m['learning_rate'].type = module_desc_pb2.FLOAT
        param_attr.optimize_attr.m['learning_rate'].f = param.optimize_attr[
            'learning_rate']
        if param.regularizer:
            if isinstance(param.regularizer,
                          fluid.regularizer.L2DecayRegularizer):
                param_attr.regularizer.type = "L2DecayRegularizer"
            if isinstance(param.regularizer,
                          fluid.regularizer.L1DecayRegularizer):
                param_attr.regularizer.type = "L1DecayRegularizer"
            param_attr.regularizer.regularization_coeff = param.regularizer.regularization_coeff

        if param.gradient_clip_attr:
            if isinstance(param.gradient_clip_attr,
                          fluid.clip.ErrorClipByValue):
                param_attr.gradient_clip_attr.max = param.gradient_clip_attr.max
                param_attr.gradient_clip_attr.min = param.gradient_clip_attr.min
                param_attr.gradient_clip_attr.type = "ErrorClipByValue"
            if isinstance(param.gradient_clip_attr,
                          fluid.clip.GradientClipByValue):
                param_attr.gradient_clip_attr.max = param.gradient_clip_attr.max
                param_attr.gradient_clip_attr.min = param.gradient_clip_attr.min
                param_attr.gradient_clip_attr.type = "GradientClipByValue"
            if isinstance(param.gradient_clip_attr,
                          fluid.clip.GradientClipByNorm):
                param_attr.gradient_clip_attr.clip_norm = param.gradient_clip_attr.clip_norm
                param_attr.gradient_clip_attr.type = "GradientClipByNorm"
            if isinstance(param.gradient_clip_attr,
                          fluid.clip.GradientClipByGlobalNorm):
                param_attr.gradient_clip_attr.clip_norm = param.gradient_clip_attr.clip_norm
                param_attr.gradient_clip_attr.group_name = param.gradient_clip_attr.group_name
                param_attr.gradient_clip_attr.type = "GradientClipByGlobalNorm"

    # save signarture info
    sign_map = module_desc.sign2var
    program = sign_arr[0].get_inputs()[0].block.program
    for sign in sign_arr:
        if sign.get_name() in sign_map:
            raise "Error! sign_arr contains repeat signatrue %s" % sign

        var = sign_map[sign.get_name()]
        feed_desc = var.feed_desc
        fetch_desc = var.fetch_desc
        feed_names = sign.get_feed_names()
        fetch_names = sign.get_fetch_names()
        for index, input in enumerate(sign.get_inputs()):
            feed_var = feed_desc.add()
            feed_var.var_name = input.name
            feed_var.alias = feed_names[index]

        for index, output in enumerate(sign.get_outputs()):
            fetch_var = fetch_desc.add()
            fetch_var.var_name = output.name
            fetch_var.alias = fetch_names[index]

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

        with open(os.path.join(save_model_dir, "__model__"), "rb") as file:
            program_desc_str = file.read()
            rename_program = fluid.framework.Program.parse_from_string(
                program_desc_str)
            varlist = {
                var: block
                for block in rename_program.blocks for var in block.vars
                if HUB_VAR_PREFIX not in var
            }
            for var, block in varlist.items():
                old_name = var
                new_name = HUB_VAR_PREFIX + old_name
                block._rename_var(old_name, new_name)
            mkdir(save_model_dir)
            with open(os.path.join(save_model_dir, "__model__"), "wb") as f:
                f.write(rename_program.desc.serialize_to_string())

            for file in os.listdir(save_model_dir):
                if (file == "__model__" or HUB_VAR_PREFIX in file):
                    continue
                os.rename(
                    os.path.join(save_model_dir, file),
                    os.path.join(save_model_dir, HUB_VAR_PREFIX + file))

    # Serialize module_desc pb
    module_pb = module_desc.SerializeToString()
    with open(ModuleConfig.module_desc_path(module_dir), "wb") as f:
        f.write(module_pb)


class ModuleUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def connect_program(pre_program, next_program, input_dict=None):
        def _copy_vars_and_ops_in_blocks(from_block, to_block):
            for var in from_block.vars:
                var = from_block.var(var)
                var_info = copy.deepcopy(get_variable_info(var))
                if isinstance(var, fluid.framework.Parameter):
                    to_block.create_parameter(**var_info)
                else:
                    to_block.create_var(**var_info)

            for op in from_block.ops:
                op_info = {
                    'type': op.type,
                    'inputs': {
                        input: [block.var(var) for var in op.input(input)]
                        for input in op.input_names
                    },
                    'outputs': {
                        output: [block.var(var) for var in op.output(output)]
                        for output in op.output_names
                    },
                    'attrs': copy.deepcopy(op.all_attrs())
                }
                to_block.append_op(**op_info)

        assert isinstance(pre_program,
                          fluid.Program), "pre_program should be fluid.Program"
        assert isinstance(next_program,
                          fluid.Program), "next_program should be fluid.Program"
        new_program = pre_program.clone()
        if input_dict:
            assert isinstance(
                input_dict, dict
            ), "the input_dict should be a dict with string-Variable pair"
            for key, var in input_dict.items():
                assert isinstance(
                    var, fluid.framework.Variable
                ), "the input_dict should be a dict with string-Variable pair"
                var_info = copy.deepcopy(get_variable_info(var))
                input_var = new_program.global_block().create_var(**var_info)
                output_var = next_program.global_block().var(key)
                var_info = copy.deepcopy(get_variable_info(output_var))
                output_var = new_program.global_block().create_var(**var_info)
                new_program.global_block().append_op(
                    type="assign",
                    inputs={'X': input_var},
                    outputs={'Out': output_var})

        block_map = {0: 0}
        logger.info("start to connect program")
        for index, block in enumerate(next_program.blocks):
            if block.idx == 0:
                _copy_vars_and_ops_in_blocks(block, new_program.global_block())
            else:
                block_map[index] = len(new_program.blocks)
                logger.info(
                    "block_%d in next_program merge into block_%d in pre_program"
                    % (index, block_map[index]))
                new_block = new_program._create_block(
                    parent_idx=block_map[block.parent_idx])
                _copy_vars_and_ops_in_blocks(block, new_block)
        logger.info("end of connect program")
        return new_program

    @staticmethod
    def remove_feed_fetch_op(program):
        """ remove feed and fetch operator and variable for fine-tuning
        """
        logger.info("remove feed fetch op")
        block = program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(block.ops):
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            block._remove_op(index)

        # TODO(wuzewu): get feed and fetch var by other way
        block._remove_var(HUB_VAR_PREFIX + "feed")
        block._remove_var(HUB_VAR_PREFIX + "fetch")

        program.desc.flush()
