#coding:utf-8
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

import os
import time
import sys
import functools
from shutil import copyfile

import paddle
import paddle.fluid as fluid

from paddlehub.common import utils
from paddlehub.common import paddle_helper
from paddlehub.common.logger import logger
from paddlehub.common.downloader import default_downloader
from paddlehub.module import module_desc_pb2
from paddlehub.module import check_info_pb2
from paddlehub.module.signature import Signature, create_signature
from paddlehub.module.checker import ModuleChecker
from paddlehub.module.manager import default_module_manager
from paddlehub.module.base_processor import BaseProcessor
from paddlehub.io.parser import yaml_parser
from paddlehub import version

__all__ = ['Module', 'create_module']

# PaddleHub module dir name
ASSETS_DIRNAME = "assets"
MODEL_DIRNAME = "model"
MODULE_DESC_PBNAME = "module_desc.pb"
PYTHON_DIR = "python"
PROCESSOR_NAME = "processor"
# PaddleHub var prefix
HUB_VAR_PREFIX = "@HUB_%s@"


def create_module(sign_arr,
                  module_dir,
                  processor=None,
                  assets=None,
                  module_info=None,
                  exe=None,
                  extra_info=None):
    sign_arr = utils.to_list(sign_arr)
    module = Module(
        signatures=sign_arr,
        processor=processor,
        assets=assets,
        module_info=module_info,
        extra_info=extra_info)
    module.serialize_to_path(path=module_dir, exe=exe)


class ModuleHelper(object):
    def __init__(self, module_dir):
        self.module_dir = module_dir

    def module_desc_path(self):
        return os.path.join(self.module_dir, MODULE_DESC_PBNAME)

    def model_path(self):
        return os.path.join(self.module_dir, MODEL_DIRNAME)

    def processor_path(self):
        return os.path.join(self.module_dir, PYTHON_DIR)

    def processor_name(self):
        return PROCESSOR_NAME

    def assets_path(self):
        return os.path.join(self.module_dir, ASSETS_DIRNAME)


class Module(object):
    def __init__(self,
                 name=None,
                 module_dir=None,
                 signatures=None,
                 module_info=None,
                 assets=None,
                 processor=None,
                 extra_info=None,
                 version=None):
        self.desc = module_desc_pb2.ModuleDesc()
        self.program = None
        self.assets = []
        self.helper = None
        self.signatures = {}
        self.default_signature = None
        self.module_info = None
        self.processor = None
        self.extra_info = {} if extra_info is None else extra_info
        if not isinstance(self.extra_info, dict):
            raise TypeError(
                "The extra_info should be an instance of python dict")

        # cache data
        self.last_call_name = None
        self.cache_feed_dict = None
        self.cache_fetch_dict = None
        self.cache_program = None

        # TODO(wuzewu): print more module loading info log
        if name:
            self._init_with_name(name=name, version=version)
        elif module_dir:
            self._init_with_module_file(module_dir=module_dir[0])
        elif signatures:
            if processor:
                if not issubclass(processor, BaseProcessor):
                    raise TypeError(
                        "Processor shoule be an instance of paddlehub.BaseProcessor"
                    )
            if assets:
                self.assets = utils.to_list(assets)
                # for asset in assets:
                #     utils.check_path(assets)
            self.processor = processor
            self._generate_module_info(module_info)
            self._init_with_signature(signatures=signatures)
        else:
            raise ValueError("Module initialized parameter is empty")

    def _init_with_name(self, name, version=None):
        log_msg = "Installing %s module" % name
        if version:
            log_msg += "-%s" % version
        logger.info(log_msg)
        result, tips, module_dir = default_module_manager.install_module(
            module_name=name, module_version=version)
        if not result:
            logger.error(tips)
            exit(1)
        logger.info(tips)
        self._init_with_module_file(module_dir[0])

    def _init_with_url(self, url):
        utils.check_url(url)
        result, tips, module_dir = default_downloader.download_file_and_uncompress(
            url, save_path=".")
        if not result:
            logger.error(tips)
            exit(1)
        self._init_with_module_file(module_dir)

    def _dump_processor(self):
        import inspect
        pymodule = inspect.getmodule(self.processor)
        pycode = inspect.getsource(pymodule)
        processor_path = self.helper.processor_path()
        processor_md5 = utils.md5(pycode)
        processor_md5 += str(time.time())
        processor_name = utils.md5(processor_md5)
        output_file = os.path.join(processor_path, processor_name + ".py")
        utils.mkdir(processor_path)
        with open(output_file, "w") as file:
            file.write(pycode)
        utils.from_pyobj_to_module_attr(
            processor_name, self.desc.attr.map.data['processor_info'])

    def _load_processor(self):
        processor_path = self.helper.processor_path()
        if os.path.exists(processor_path):
            sys.path.append(processor_path)
            processor_name = utils.from_module_attr_to_pyobj(
                self.desc.attr.map.data['processor_info'])
            self.processor = __import__(processor_name).Processor(module=self)
        else:
            self.processor = None

    def _dump_assets(self):
        utils.mkdir(self.helper.assets_path())
        for asset in self.assets:
            filename = os.path.basename(asset)
            newfile = os.path.join(self.helper.assets_path(), filename)
            copyfile(asset, newfile)

    def _load_assets(self):
        assets_path = self.helper.assets_path()
        self.assets = []
        for file in os.listdir(assets_path):
            filepath = os.path.join(self.helper.assets_path(), file)
            self.assets.append(filepath)

    def _init_with_module_file(self, module_dir):
        checker = ModuleChecker(module_dir)
        checker.check()

        self.helper = ModuleHelper(module_dir)
        with open(self.helper.module_desc_path(), "rb") as fi:
            self.desc.ParseFromString(fi.read())

        exe = fluid.Executor(fluid.CPUPlace())
        self.program, _, _ = fluid.io.load_inference_model(
            self.helper.model_path(), executor=exe)
        for block in self.program.blocks:
            for op in block.ops:
                if "op_callstack" in op.all_attrs():
                    op._set_attr("op_callstack", [""])
        self._load_processor()
        self._load_assets()
        self._recover_from_desc()
        self._generate_sign_attr()
        self._generate_extra_info()
        self._restore_parameter(self.program)
        self._recover_variable_info(self.program)

    def _init_with_signature(self, signatures):
        self.name_prefix = HUB_VAR_PREFIX % self.name
        self._process_signatures(signatures)
        self._check_signatures()
        self._generate_desc()
        self._generate_sign_attr()
        self._generate_extra_info()

    def _init_with_program(self, program):
        pass

    def _process_signatures(self, signatures):
        self.signatures = {}
        self.program = signatures[0].inputs[0].block.program
        for sign in signatures:
            if sign.name in self.signatures:
                raise ValueError(
                    "Error! Signature array contains duplicated signatrues %s" %
                    sign)
            if self.default_signature is None and sign.for_predict:
                self.default_signature = sign
            self.signatures[sign.name] = sign

    def _restore_parameter(self, program):
        global_block = program.global_block()
        param_attrs = self.desc.attr.map.data['param_attrs']
        for key, param_attr in param_attrs.map.data.items():
            param = paddle_helper.from_module_attr_to_param(param_attr)
            param['name'] = self.get_var_name_with_prefix(key)
            if (param['name'] not in global_block.vars):
                continue
            var = global_block.var(param['name'])
            global_block.create_parameter(
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                error_clip=var.error_clip,
                stop_gradient=var.stop_gradient,
                is_data=var.is_data,
                **param)

    def _recover_variable_info(self, program):
        var_infos = self.desc.attr.map.data['var_infos']
        for var_info in var_infos.map.data:
            idx = utils.from_module_attr_to_pyobj(
                var_infos.map.data[var_info].map.data['block_id'])
            stop_gradient = utils.from_module_attr_to_pyobj(
                var_infos.map.data[var_info].map.data['stop_gradient'])
            block = program.blocks[idx]
            var_name = self.get_var_name_with_prefix(var_info)
            if var_name in block.vars:
                var = block.vars[var_name]
                var.stop_gradient = stop_gradient

    def get_extra_info(self, key):
        return self.extra_info.get(key, None)

    def _generate_extra_info(self):
        for key in self.extra_info:
            self.__dict__["get_%s" % key] = functools.partial(
                self.get_extra_info, key=key)

    def _generate_module_info(self, module_info=None):
        if not module_info:
            self.module_info = {}
        else:
            if not utils.is_yaml_file(module_info):
                logger.critical("Module info file should be yaml format")
                exit(1)
            self.module_info = yaml_parser.parse(module_info)
        self.author = self.module_info.get('author', 'UNKNOWN')
        self.author_email = self.module_info.get('author_email', 'UNKNOWN')
        self.summary = self.module_info.get('summary', 'UNKNOWN')
        self.type = self.module_info.get('type', 'UNKNOWN')
        self.version = self.module_info.get('version', 'UNKNOWN')
        self.name = self.module_info.get('name', 'UNKNOWN')

    def _generate_sign_attr(self):
        self._check_signatures()
        for sign in self.signatures:
            self.__dict__[sign] = functools.partial(
                self.__call__, sign_name=sign)

    def get_vocab_path(self):
        for assets_file in self.assets:
            if "vocab.txt" in assets_file:
                return assets_file

    def _recover_from_desc(self):
        # recover signature
        for sign, module_var in self.desc.sign2var.items():
            inputs = []
            outputs = []
            feed_names = []
            fetch_names = []
            for var in module_var.feed_desc:
                variable = self.program.global_block().vars[var.var_name]
                inputs.append(variable)
                feed_names.append(var.alias)

            for var in module_var.fetch_desc:
                variable = self.program.global_block().vars[var.var_name]
                outputs.append(variable)
                fetch_names.append(var.alias)

            self.signatures[sign] = create_signature(
                sign,
                inputs=inputs,
                outputs=outputs,
                feed_names=feed_names,
                fetch_names=fetch_names)

        # recover default signature
        default_signature_name = utils.from_module_attr_to_pyobj(
            self.desc.attr.map.data['default_signature'])
        self.default_signature = self.signatures[
            default_signature_name] if default_signature_name else None

        # recover module info
        module_info = self.desc.attr.map.data['module_info']
        self.name = utils.from_module_attr_to_pyobj(
            module_info.map.data['name'])
        self.author = utils.from_module_attr_to_pyobj(
            module_info.map.data['author'])
        self.author_email = utils.from_module_attr_to_pyobj(
            module_info.map.data['author_email'])
        self.version = utils.from_module_attr_to_pyobj(
            module_info.map.data['version'])
        self.type = utils.from_module_attr_to_pyobj(
            module_info.map.data['type'])
        self.summary = utils.from_module_attr_to_pyobj(
            module_info.map.data['summary'])

        # recover extra info
        extra_info = self.desc.attr.map.data['extra_info']
        self.extra_info = {}
        for key, value in extra_info.map.data.items():
            self.extra_info[key] = utils.from_module_attr_to_pyobj(value)

        # recover name prefix
        self.name_prefix = utils.from_module_attr_to_pyobj(
            self.desc.attr.map.data["name_prefix"])

    def _generate_desc(self):
        # save fluid Parameter
        attr = self.desc.attr
        attr.type = module_desc_pb2.MAP
        param_attrs = attr.map.data['param_attrs']
        param_attrs.type = module_desc_pb2.MAP
        for param in self.program.global_block().iter_parameters():
            param_attr = param_attrs.map.data[param.name]
            paddle_helper.from_param_to_module_attr(param, param_attr)

        # save Variable Info
        var_infos = attr.map.data['var_infos']
        var_infos.type = module_desc_pb2.MAP
        for block in self.program.blocks:
            for var in block.vars.values():
                var_info = var_infos.map.data[var.name]
                var_info.type = module_desc_pb2.MAP
                utils.from_pyobj_to_module_attr(
                    var.stop_gradient, var_info.map.data['stop_gradient'])
                utils.from_pyobj_to_module_attr(block.idx,
                                                var_info.map.data['block_id'])

        # save signarture info
        for key, sign in self.signatures.items():
            var = self.desc.sign2var[sign.name]
            feed_desc = var.feed_desc
            fetch_desc = var.fetch_desc
            feed_names = sign.feed_names
            fetch_names = sign.fetch_names
            for index, input in enumerate(sign.inputs):
                feed_var = feed_desc.add()
                feed_var.var_name = self.get_var_name_with_prefix(input.name)
                feed_var.alias = feed_names[index]

            for index, output in enumerate(sign.outputs):
                fetch_var = fetch_desc.add()
                fetch_var.var_name = self.get_var_name_with_prefix(output.name)
                fetch_var.alias = fetch_names[index]

        # save default signature
        utils.from_pyobj_to_module_attr(
            self.default_signature.name if self.default_signature else None,
            attr.map.data['default_signature'])

        # save name prefix
        utils.from_pyobj_to_module_attr(self.name_prefix,
                                        self.desc.attr.map.data["name_prefix"])

        # save module info
        module_info = attr.map.data['module_info']
        module_info.type = module_desc_pb2.MAP
        utils.from_pyobj_to_module_attr(self.name, module_info.map.data['name'])
        utils.from_pyobj_to_module_attr(self.version,
                                        module_info.map.data['version'])
        utils.from_pyobj_to_module_attr(self.author,
                                        module_info.map.data['author'])
        utils.from_pyobj_to_module_attr(self.author_email,
                                        module_info.map.data['author_email'])
        utils.from_pyobj_to_module_attr(self.type, module_info.map.data['type'])
        utils.from_pyobj_to_module_attr(self.summary,
                                        module_info.map.data['summary'])

        # save extra info
        extra_info = attr.map.data['extra_info']
        extra_info.type = module_desc_pb2.MAP
        for key, value in self.extra_info.items():
            utils.from_pyobj_to_module_attr(value, extra_info.map.data[key])

    def __call__(self, sign_name, data, use_gpu=False, batch_size=1, **kwargs):
        self.check_processor()

        def _get_reader_and_feeder(data_format, data, place):
            def _reader(process_data):
                for item in zip(*process_data):
                    yield item

            process_data = []
            feed_name_list = []
            for key in data_format:
                process_data.append([value['processed'] for value in data[key]])
                feed_name_list.append(data_format[key]['feed_key'])
            feeder = fluid.DataFeeder(feed_list=feed_name_list, place=place)
            return functools.partial(_reader, process_data=process_data), feeder

        if self.last_call_name != sign_name:
            self.last_call_name = sign_name
            self.cache_feed_dict, self.cache_fetch_dict, self.cache_program = self.context(
                sign_name, for_test=True)
        feed_dict = self.cache_feed_dict
        fetch_dict = self.cache_fetch_dict
        program = self.cache_program

        #TODO(wuzewu): more option
        fetch_list = list(set([value for key, value in fetch_dict.items()]))
        with fluid.program_guard(program):
            result = []
            index = 0
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                use_gpu = False

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

            exe = fluid.Executor(place=place)
            data = self.processor.preprocess(
                sign_name=sign_name, data_dict=data)
            data_format = self.processor.data_format(sign_name=sign_name)
            reader, feeder = _get_reader_and_feeder(data_format, data, place)
            reader = paddle.batch(reader, batch_size=batch_size)
            for batch in reader():
                data_out = exe.run(
                    feed=feeder.feed(batch),
                    fetch_list=fetch_list,
                    return_numpy=False)
                sub_data = {
                    key: value[index:index + len(batch)]
                    for key, value in data.items()
                }
                result += self.processor.postprocess(sign_name, data_out,
                                                     sub_data, **kwargs)
                index += len(batch)

        return result

    def check_processor(self):
        if not self.processor:
            raise ValueError("This Module is not callable!")

    def context(self,
                sign_name=None,
                for_test=False,
                trainable=True,
                regularizer=None,
                max_seq_len=128,
                learning_rate=1e-3):
        """
        Args:
            max_seq_len(int): maximum sequence length, this option is only
            available for BERT/ERNIE module
        """

        if sign_name:
            if sign_name not in self.signatures:
                raise KeyError(
                    "Module did not have a signature with name %s" % sign_name)
            signature = self.signatures[sign_name]
        else:
            inputs = [
                input for signature in self.signatures.values()
                for input in signature.inputs
            ]
            outputs = [
                output for signature in self.signatures.values()
                for output in signature.outputs
            ]
            feed_names = [
                feed_name for signature in self.signatures.values()
                for feed_name in signature.feed_names
            ]
            fetch_names = [
                fetch_name for signature in self.signatures.values()
                for fetch_name in signature.fetch_names
            ]
            signature = create_signature(
                name="hub_temp_signature",
                inputs=inputs,
                outputs=outputs,
                feed_names=feed_names,
                fetch_names=fetch_names,
                for_predict=False)

        program = self.program.clone(for_test=for_test)
        paddle_helper.remove_feed_fetch_op(program)

        if not for_test:
            paddle_helper.set_parameter_trainable(program, trainable)

            paddle_helper.set_parameter_learning_rate(program, learning_rate)

            paddle_helper.set_parameter_regularizer(program, regularizer)

            self._restore_parameter(program)

        self._recover_variable_info(program)

        paddle_helper.set_op_attr(program, is_test=for_test)
        #TODO(wuzewu): return feed_list and fetch_list directly
        feed_dict = {}
        fetch_dict = {}
        for index, var in enumerate(signature.inputs):
            feed_dict[index] = program.global_block().var(var.name)
            key = signature.feed_names[index]
            if key:
                feed_dict[key] = program.global_block().var(var.name)

        for index, var in enumerate(signature.outputs):
            fetch_dict[index] = program.global_block().var(var.name)
            key = signature.fetch_names[index]
            if key:
                fetch_dict[key] = program.global_block().var(var.name)

        # TODO(ZeyuChen) encapsulate into a funtion
        # update BERT/ERNIE's input tensor's sequence length to max_seq_len
        if self.name.startswith("bert") or self.name.startswith("ernie"):
            MAX_SEQ_LENGTH = 512
            if max_seq_len > MAX_SEQ_LENGTH or max_seq_len <= 0:
                raise ValueError(
                    "max_seq_len({}) should be in the range of [1, {}]".format(
                        MAX_SEQ_LENGTH))
            logger.info(
                "Set maximum sequence length of input tensor to {}".format(
                    max_seq_len))
            if self.name.startswith("ernie_v2"):
                feed_list = [
                    "input_ids", "position_ids", "segment_ids", "input_mask",
                    "task_ids"
                ]
                logger.warning(
                    "%s will exploite task_id, the arguement use_taskid of Reader class must be True."
                    % self.name)
            else:
                feed_list = [
                    "input_ids", "position_ids", "segment_ids", "input_mask"
                ]
                logger.warning(
                    "%s has no task_id, the arguement use_taskid of Reader class must be False."
                    % self.name)
            for tensor_name in feed_list:
                seq_tensor_shape = [-1, max_seq_len, 1]
                logger.info("The shape of input tensor[{}] set to {}".format(
                    tensor_name, seq_tensor_shape))
                program.global_block().var(
                    feed_dict[tensor_name].name).desc.set_shape(
                        seq_tensor_shape)

        # record num parameters loaded by paddlehub
        num_param_loaded = 0
        for param in program.global_block().iter_parameters():
            num_param_loaded += 1
        logger.info(
            "%d pretrained paramaters loaded by PaddleHub" % num_param_loaded)

        return feed_dict, fetch_dict, program

    def get_name_prefix(self):
        return self.name_prefix

    def get_var_name_with_prefix(self, var_name):
        return self.get_name_prefix() + var_name

    def _check_signatures(self):
        if not self.signatures:
            raise ValueError("Signatures should not be None")

        for key, sign in self.signatures.items():
            if not isinstance(sign, Signature):
                raise TypeError(
                    "Item in Signatures shoule be an instance of paddlehub.Signature"
                )

            for input in sign.inputs:
                _tmp_program = input.block.program
                if not self.program == _tmp_program:
                    raise ValueError(
                        "All input and outputs variables in signature should come from the same Program"
                    )

            for output in sign.outputs:
                _tmp_program = output.block.program
                if not self.program == _tmp_program:
                    raise ValueError(
                        "All input and outputs variables in signature should come from the same Program"
                    )

    def serialize_to_path(self, path=None, exe=None):
        self._check_signatures()
        self._generate_desc()
        # create module path for saving
        if path is None:
            path = os.path.join(".", self.name)
        self.helper = ModuleHelper(path)
        utils.mkdir(self.helper.module_dir)

        # create module pb
        module_desc = module_desc_pb2.ModuleDesc()
        logger.info("PaddleHub version = %s" % version.hub_version)
        logger.info("PaddleHub Module proto version = %s" %
                    version.module_proto_version)
        logger.info("Paddle version = %s" % paddle.__version__)

        feeded_var_names = [
            input.name for key, sign in self.signatures.items()
            for input in sign.inputs
        ]
        target_vars = [
            output for key, sign in self.signatures.items()
            for output in sign.outputs
        ]
        feeded_var_names = list(set(feeded_var_names))
        target_vars = list(set(target_vars))

        # save inference program
        program = self.program.clone()

        for block in program.blocks:
            for op in block.ops:
                if "op_callstack" in op.all_attrs():
                    op._set_attr("op_callstack", [""])

        if not exe:
            place = fluid.CPUPlace()
            exe = fluid.Executor(place=place)
        utils.mkdir(self.helper.model_path())
        fluid.io.save_inference_model(
            self.helper.model_path(),
            feeded_var_names=list(feeded_var_names),
            target_vars=list(target_vars),
            main_program=program,
            executor=exe)

        with open(os.path.join(self.helper.model_path(), "__model__"),
                  "rb") as file:
            program_desc_str = file.read()
            rename_program = fluid.framework.Program.parse_from_string(
                program_desc_str)
            varlist = {
                var: block
                for block in rename_program.blocks for var in block.vars
                if self.get_name_prefix() not in var
            }
            for var, block in varlist.items():
                old_name = var
                new_name = self.get_var_name_with_prefix(old_name)
                block._rename_var(old_name, new_name)
            utils.mkdir(self.helper.model_path())
            with open(
                    os.path.join(self.helper.model_path(), "__model__"),
                    "wb") as f:
                f.write(rename_program.desc.serialize_to_string())

            for file in os.listdir(self.helper.model_path()):
                if (file == "__model__" or self.get_name_prefix() in file):
                    continue
                os.rename(
                    os.path.join(self.helper.model_path(), file),
                    os.path.join(self.helper.model_path(),
                                 self.get_var_name_with_prefix(file)))

        # create processor file
        if self.processor:
            self._dump_processor()

        # create assets
        self._dump_assets()

        # create check info
        checker = ModuleChecker(self.helper.module_dir)
        checker.generate_check_info()

        # Serialize module_desc pb
        module_pb = self.desc.SerializeToString()
        with open(self.helper.module_desc_path(), "wb") as f:
            f.write(module_pb)
