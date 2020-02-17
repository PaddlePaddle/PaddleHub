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
import inspect
import importlib
import tarfile
import six
import shutil

import paddle
import paddle.fluid as fluid

from paddlehub.common import utils
from paddlehub.common import paddle_helper
from paddlehub.common.dir import CACHE_HOME
from paddlehub.common.lock import lock
from paddlehub.common.logger import logger
from paddlehub.common.hub_server import CacheUpdater
from paddlehub.common import tmp_dir
from paddlehub.common.downloader import progress
from paddlehub.module import module_desc_pb2
from paddlehub.module.manager import default_module_manager
from paddlehub.module.checker import ModuleChecker
from paddlehub.module.signature import Signature, create_signature
from paddlehub.module.base_processor import BaseProcessor
from paddlehub.io.parser import yaml_parser
from paddlehub import version

# PaddleHub module dir name
ASSETS_DIRNAME = "assets"
MODEL_DIRNAME = "model"
MODULE_DESC_PBNAME = "module_desc.pb"
PYTHON_DIR = "python"
PROCESSOR_NAME = "processor"
# PaddleHub var prefix
HUB_VAR_PREFIX = "@HUB_%s@"
# PaddleHub Module package suffix
HUB_PACKAGE_SUFFIX = "phm"


def create_module(directory, name, author, email, module_type, summary,
                  version):
    save_file = "{}-{}.{}".format(name, version, HUB_PACKAGE_SUFFIX)

    with tmp_dir() as base_dir:
        # package the module
        with tarfile.open(save_file, "w:gz") as tar:
            module_dir = os.path.join(base_dir, name)
            shutil.copytree(directory, module_dir)

            # record module info and serialize
            desc = module_desc_pb2.ModuleDesc()
            attr = desc.attr
            attr.type = module_desc_pb2.MAP
            module_info = attr.map.data['module_info']
            module_info.type = module_desc_pb2.MAP
            utils.from_pyobj_to_module_attr(name, module_info.map.data['name'])
            utils.from_pyobj_to_module_attr(author,
                                            module_info.map.data['author'])
            utils.from_pyobj_to_module_attr(
                email, module_info.map.data['author_email'])
            utils.from_pyobj_to_module_attr(module_type,
                                            module_info.map.data['type'])
            utils.from_pyobj_to_module_attr(summary,
                                            module_info.map.data['summary'])
            utils.from_pyobj_to_module_attr(version,
                                            module_info.map.data['version'])
            module_desc_path = os.path.join(module_dir, "module_desc.pb")
            with open(module_desc_path, "wb") as f:
                f.write(desc.SerializeToString())

            # generate check info
            checker = ModuleChecker(module_dir)
            checker.generate_check_info()

            # add __init__
            module_init = os.path.join(module_dir, "__init__.py")
            with open(module_init, "a") as file:
                file.write("")

            _cwd = os.getcwd()
            os.chdir(base_dir)
            module_dir = module_dir.replace(base_dir, ".")
            tar.add(module_dir, recursive=False)
            files = []
            for dirname, _, subfiles in os.walk(module_dir):
                for file in subfiles:
                    #                     if file.startswith("."):
                    #                         continue
                    files.append(os.path.join(dirname, file))

            total_length = len(files)
            print("Create Module {}-{}".format(name, version))
            for index, file in enumerate(files):
                done = int(float(index) / total_length * 50)
                progress("[%-50s] %.2f%%" % ('=' * done,
                                             float(index / total_length * 100)))
                tar.add(file)
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)
            print("Module package saved as {}".format(save_file))
            os.chdir(_cwd)


_module_runnable_func = {}


def runnable(func):
    mod = func.__module__ + "." + inspect.stack()[1][3]
    _module_runnable_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


class Module(object):

    _record = {}

    def __new__(cls, name=None, directory=None, module_dir=None, version=None):
        if cls.__name__ == "Module":
            if name:
                module = cls.init_with_name(name=name, version=version)
            elif directory:
                module = cls.init_with_directory(directory=directory)
            elif module_dir:
                logger.warning(
                    "Parameter module_dir is deprecated, please use directory to specify the path"
                )
                if isinstance(module_dir, list) or isinstance(
                        module_dir, tuple):
                    directory = module_dir[0]
                    version = module_dir[1]
                else:
                    directory = module_dir
                module = cls.init_with_directory(directory=directory)
            CacheUpdater("update_cache", module.name, module.version).start()
        else:
            module = object.__new__(cls)

        return module

    def __init__(self, name=None, directory=None, module_dir=None,
                 version=None):
        # Avoid module being initialized multiple times
        if not directory or id(self) in Module._record:
            return
        Module._record[id(self)] = True

        mod = self.__class__.__module__ + "." + self.__class__.__name__
        if mod in _module_runnable_func:
            _run_func_name = _module_runnable_func[mod]
            self._run_func = getattr(self, _run_func_name)
        else:
            self._run_func = None
        self._code_version = "v2"
        self._directory = directory
        self.module_desc_path = os.path.join(self.directory, MODULE_DESC_PBNAME)
        self._desc = module_desc_pb2.ModuleDesc()
        with open(self.module_desc_path, "rb") as file:
            self._desc.ParseFromString(file.read())

        module_info = self.desc.attr.map.data['module_info']
        self._name = utils.from_module_attr_to_pyobj(
            module_info.map.data['name'])
        self._author = utils.from_module_attr_to_pyobj(
            module_info.map.data['author'])
        self._author_email = utils.from_module_attr_to_pyobj(
            module_info.map.data['author_email'])
        self._version = utils.from_module_attr_to_pyobj(
            module_info.map.data['version'])
        self._type = utils.from_module_attr_to_pyobj(
            module_info.map.data['type'])
        self._summary = utils.from_module_attr_to_pyobj(
            module_info.map.data['summary'])

        self._initialize()

    @classmethod
    def init_with_name(cls, name, version=None):
        fp_lock = open(os.path.join(CACHE_HOME, name), "a")
        lock.flock(fp_lock, lock.LOCK_EX)
        log_msg = "Installing %s module" % name
        if version:
            log_msg += "-%s" % version
        logger.info(log_msg)
        extra = {"command": "install"}
        result, tips, module_dir = default_module_manager.install_module(
            module_name=name, module_version=version, extra=extra)
        if not result:
            logger.error(tips)
            raise RuntimeError(tips)

        logger.info(tips)
        lock.flock(fp_lock, lock.LOCK_UN)
        return cls.init_with_directory(directory=module_dir[0])

    @classmethod
    def init_with_directory(cls, directory):
        desc_file = os.path.join(directory, MODULE_DESC_PBNAME)
        checker = ModuleChecker(directory)
        checker.check()

        module_code_version = checker.module_code_version
        if module_code_version == "v2":
            basename = os.path.split(directory)[-1]
            dirname = os.path.join(*list(os.path.split(directory)[:-1]))
            sys.path.append(dirname)
            user_module = importlib.import_module("{}.module".format(basename))
            return user_module.HubModule(directory=directory)
        return ModuleV1(directory=directory)

    @property
    def run_func(self):
        return self._run_func

    @property
    def desc(self):
        return self._desc

    @property
    def directory(self):
        return self._directory

    @property
    def author(self):
        return self._author

    @property
    def author_email(self):
        return self._author_email

    @property
    def summary(self):
        return self._summary

    @property
    def type(self):
        return self._type

    @property
    def version(self):
        return self._version

    @property
    def name(self):
        return self._name

    @property
    def code_version(self):
        return self._code_version

    @property
    def is_runnable(self):
        return self._run_func != None

    def _initialize(self):
        pass


class ModuleHelper(object):
    def __init__(self, directory):
        self.directory = directory

    def module_desc_path(self):
        return os.path.join(self.directory, MODULE_DESC_PBNAME)

    def model_path(self):
        return os.path.join(self.directory, MODEL_DIRNAME)

    def processor_path(self):
        return os.path.join(self.directory, PYTHON_DIR)

    def processor_name(self):
        return PROCESSOR_NAME

    def assets_path(self):
        return os.path.join(self.directory, ASSETS_DIRNAME)


class ModuleV1(Module):
    def __init__(self, name=None, directory=None, module_dir=None,
                 version=None):
        if not directory:
            return
        super(ModuleV1, self).__init__(name, directory, module_dir, version)
        self._code_version = "v1"
        self.program = None
        self.assets = []
        self.helper = None
        self.signatures = {}
        self.default_signature = None
        self.processor = None
        self.extra_info = {}

        # cache data
        self.last_call_name = None
        self.cache_feed_dict = None
        self.cache_fetch_dict = None
        self.cache_program = None

        self.helper = ModuleHelper(directory)
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
            shutil.copyfile(asset, newfile)

    def _load_assets(self):
        assets_path = self.helper.assets_path()
        self.assets = []
        for file in os.listdir(assets_path):
            filepath = os.path.join(self.helper.assets_path(), file)
            self.assets.append(filepath)

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

    def _generate_sign_attr(self):
        self._check_signatures()
        for sign in self.signatures:
            self.__dict__[sign] = functools.partial(
                self.__call__, sign_name=sign)

    def get_vocab_path(self):
        for assets_file in self.assets:
            if "vocab.txt" in assets_file:
                return assets_file
        return None

    def get_word_dict_path(self):
        for assets_file in self.assets:
            if "dict.wordseg.pickle" in assets_file:
                return assets_file
        return None

    def get_spm_path(self):
        for assets_file in self.assets:
            if "spm_cased_simp_sampled.model" in assets_file:
                return assets_file
        return None

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
            default_signature_name].name if default_signature_name else None

        # recover module info
        module_info = self.desc.attr.map.data['module_info']
        self._name = utils.from_module_attr_to_pyobj(
            module_info.map.data['name'])
        self._author = utils.from_module_attr_to_pyobj(
            module_info.map.data['author'])
        self._author_email = utils.from_module_attr_to_pyobj(
            module_info.map.data['author_email'])
        self._version = utils.from_module_attr_to_pyobj(
            module_info.map.data['version'])
        self._type = utils.from_module_attr_to_pyobj(
            module_info.map.data['type'])
        self._summary = utils.from_module_attr_to_pyobj(
            module_info.map.data['summary'])

        # recover extra info
        extra_info = self.desc.attr.map.data['extra_info']
        self.extra_info = {}
        for key, value in extra_info.map.data.items():
            self.extra_info[key] = utils.from_module_attr_to_pyobj(value)

        # recover name prefix
        self._name_prefix = utils.from_module_attr_to_pyobj(
            self.desc.attr.map.data["name_prefix"])

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

    @property
    def is_runnable(self):
        return self.default_signature != None

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

        # update BERT/ERNIE's input tensor's sequence length to max_seq_len
        if "bert" in self.name or self.name.startswith("ernie"):
            MAX_SEQ_LENGTH = 512
            if max_seq_len > MAX_SEQ_LENGTH or max_seq_len <= 0:
                raise ValueError(
                    "max_seq_len({}) should be in the range of [1, {}]".format(
                        max_seq_len, MAX_SEQ_LENGTH))
            logger.info(
                "Set maximum sequence length of input tensor to {}".format(
                    max_seq_len))
            if self.name.startswith("ernie_v2"):
                feed_list = [
                    "input_ids", "position_ids", "segment_ids", "input_mask",
                    "task_ids"
                ]
            else:
                feed_list = [
                    "input_ids", "position_ids", "segment_ids", "input_mask"
                ]
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
        return self._name_prefix

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
