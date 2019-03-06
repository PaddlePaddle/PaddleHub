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
from paddle_hub.tools import utils
from paddle_hub.tools.logger import logger
from paddle_hub.tools import downloader
from paddle_hub.tools import paddle_helper
from paddle_hub.module import module_desc_pb2
from paddle_hub.module.signature import Signature, create_signature
from paddle_hub.data.reader import yaml_reader
from paddle_hub import version
from paddle_hub.module.base_processor import BaseProcessor
from shutil import copyfile
import os
import functools
import paddle
import paddle.fluid as fluid

__all__ = ['Module', 'create_module']


def create_module(sign_arr,
                  module_dir,
                  processor,
                  assets=None,
                  module_info=None,
                  exe=None):
    sign_arr = utils.to_list(sign_arr)
    module = Module(
        signatures=sign_arr,
        processor=processor,
        assets=assets,
        module_info=module_info)
    module.serialize_to_path(path=module_dir, exe=exe)


# paddle hub module dir name
ASSETS_DIRNAME = "assets"
MODEL_DIRNAME = "model"
MODULE_DESC_PBNAME = "module_desc.pb"
PYTHON_DIR = "python"
PROCESSOR_NAME = "processor"
# paddle hub var prefix
HUB_VAR_PREFIX = "@HUB@"


class ModuleHelper:
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


class Module:
    def __init__(self,
                 url=None,
                 module_dir=None,
                 signatures=None,
                 module_info=None,
                 assets=None,
                 processor=None):
        self.desc = module_desc_pb2.ModuleDesc()
        self.program = None
        self.assets = []
        self.helper = None
        self.signatures = {}
        self.default_signature = None
        self.module_info = None
        self.processor = None
        self.assets = []
        if url:
            self._init_with_url(url=url)
        elif module_dir:
            self._init_with_module_file(module_dir=module_dir)
        elif signatures:
            assert processor, "lack of module processor"
            assert issubclass(
                processor, BaseProcessor
            ), "processor should be sub class of hub.BaseProcessor"
            if assets:
                self.assets = utils.to_list(assets)
                for asset in assets:
                    utils.check_path(assets)
            self.processor = processor
            self._generate_module_info(module_info)
            self._init_with_signature(signatures=signatures)
        else:
            raise "Error! HubModule Can't init with nothing"

    def _init_with_url(self, url):
        utils.check_url_valid(url)
        module_dir = downloader.download_and_uncompress(module_url)
        self._init_with_module_file(module_dir)

    def _dump_processor(self):
        import inspect
        pymodule = inspect.getmodule(self.processor)
        pycode = inspect.getsource(pymodule)
        processor_path = self.helper.processor_path()
        processor_name = self.helper.processor_name()
        output_file = os.path.join(processor_path, processor_name + ".py")
        utils.mkdir(processor_path)
        with open(output_file, "w") as file:
            file.write(pycode)

    def _load_processor(self):
        import sys
        processor_path = self.helper.processor_path()
        sys.path.append(processor_path)
        processor_name = self.helper.processor_name()
        self.processor = __import__(processor_name).Processor(module=self)

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
        self.helper = ModuleHelper(module_dir)
        with open(self.helper.module_desc_path(), "rb") as fi:
            self.desc.ParseFromString(fi.read())

        exe = fluid.Executor(fluid.CPUPlace())
        self.program, _, _ = fluid.io.load_inference_model(
            self.helper.model_path(), executor=exe)
        self._recovery_parameter(self.program)
        self._recover_variable_info(self.program)
        self._load_processor()
        self._load_assets()
        self._recover_from_desc()
        self._generate_sign_attr()

    def _init_with_signature(self, signatures):
        self._process_signatures(signatures)
        self._check_signatures()
        self._generate_desc()
        self._generate_sign_attr()

    def _init_with_program(self, program):
        pass

    def _process_signatures(self, signatures):
        self.signatures = {}
        self.program = signatures[0].inputs[0].block.program
        for sign in signatures:
            if sign.name in self.signatures:
                raise "Error! signature array contains repeat signatrue %s" % sign
            self.signatures[sign.name] = sign

    def _recovery_parameter(self, program):
        global_block = self.program.global_block()
        param_attrs = self.desc.extra_info.map.data['param_attrs']
        for key, param_attr in param_attrs.map.data.items():
            param = paddle_helper.from_flexible_data_to_param(param_attr)
            param['name'] = HUB_VAR_PREFIX + key
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

    def _recover_variable_info(self, program):
        var_infos = self.desc.extra_info.map.data['var_infos']
        for var_info in var_infos.map.data:
            idx = utils.from_flexible_data_to_pyobj(
                var_infos.map.data[var_info].map.data['block_id'])
            stop_gradient = utils.from_flexible_data_to_pyobj(
                var_infos.map.data[var_info].map.data['stop_gradient'])
            block = program.blocks[idx]
            var_name = HUB_VAR_PREFIX + var_info
            if var_name in block.vars:
                var = block.vars[var_name]
                var.stop_gradient = stop_gradient

    def _generate_module_info(self, module_info=None):
        if not module_info:
            self.module_info = {}
        else:
            if not utils.is_yaml_file(module_info):
                logger.critical("module info file should in yaml format")
                exit(1)
            module_info = yaml_reader.read(module_info)
        self.author = module_info.get('author', 'UNKNOWN')
        self.author_email = module_info.get('author_email', 'UNKNOWN')
        self.summary = module_info.get('summary', 'UNKNOWN')
        self.type = module_info.get('type', 'UNKNOWN')
        self.version = module_info.get('version', 'UNKNOWN')
        self.name = module_info.get('name', 'UNKNOWN')

    def _generate_sign_attr(self):
        self._check_signatures()
        for sign in self.signatures:
            self.__dict__[sign] = functools.partial(
                self.__call__, sign_name=sign)

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

        # recover module info
        module_info = self.desc.extra_info.map.data['module_info']
        self.name = utils.from_flexible_data_to_pyobj(
            module_info.map.data['name'])
        self.author = utils.from_flexible_data_to_pyobj(
            module_info.map.data['author'])
        self.author_email = utils.from_flexible_data_to_pyobj(
            module_info.map.data['author_email'])
        self.version = utils.from_flexible_data_to_pyobj(
            module_info.map.data['version'])
        self.type = utils.from_flexible_data_to_pyobj(
            module_info.map.data['type'])
        self.summary = utils.from_flexible_data_to_pyobj(
            module_info.map.data['summary'])

    def _generate_desc(self):
        # save fluid Parameter
        extra_info = self.desc.extra_info
        extra_info.type = module_desc_pb2.MAP
        param_attrs = extra_info.map.data['param_attrs']
        param_attrs.type = module_desc_pb2.MAP
        for param in self.program.global_block().iter_parameters():
            param_attr = param_attrs.map.data[param.name]
            paddle_helper.from_param_to_flexible_data(param, param_attr)

        # save Variable Info
        var_infos = extra_info.map.data['var_infos']
        var_infos.type = module_desc_pb2.MAP
        for block in self.program.blocks:
            for var in block.vars.values():
                var_info = var_infos.map.data[var.name]
                var_info.type = module_desc_pb2.MAP
                utils.from_pyobj_to_flexible_data(
                    var.stop_gradient, var_info.map.data['stop_gradient'])
                utils.from_pyobj_to_flexible_data(block.idx,
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
                feed_var.var_name = HUB_VAR_PREFIX + input.name
                feed_var.alias = feed_names[index]

            for index, output in enumerate(sign.outputs):
                fetch_var = fetch_desc.add()
                fetch_var.var_name = HUB_VAR_PREFIX + output.name
                fetch_var.alias = fetch_names[index]

        # save module info
        module_info = extra_info.map.data['module_info']
        module_info.type = module_desc_pb2.MAP
        utils.from_pyobj_to_flexible_data(self.name,
                                          module_info.map.data['name'])
        utils.from_pyobj_to_flexible_data(self.version,
                                          module_info.map.data['version'])
        utils.from_pyobj_to_flexible_data(self.author,
                                          module_info.map.data['author'])
        utils.from_pyobj_to_flexible_data(self.author_email,
                                          module_info.map.data['author_email'])
        utils.from_pyobj_to_flexible_data(self.type,
                                          module_info.map.data['type'])
        utils.from_pyobj_to_flexible_data(self.summary,
                                          module_info.map.data['summary'])

    def __call__(self, sign_name, data, config=None):
        feed_dict, fetch_dict, program = self.context(sign_name)
        #TODO(wuzewu): more option
        program = program.clone(for_test=True)
        reader = self.processor.reader(sign_name=sign_name, data_dict=data)
        feed_name_list = list(
            set([value.name for key, value in feed_dict.items()]))
        fetch_list = list(set([value for key, value in fetch_dict.items()]))
        with fluid.program_guard(program):
            place = fluid.CPUPlace()
            exe = fluid.Executor(place=place)
            feeder = fluid.DataFeeder(feed_list=feed_name_list, place=place)
            for batch in reader():
                data_out = exe.run(
                    feed=feeder.feed(batch),
                    fetch_list=fetch_list,
                    return_numpy=False)
                self.processor.postprocess(sign_name, data_out, config)

    def context(self, sign_name, trainable=False):

        assert sign_name in self.signatures, "module did not have a signature with name %s" % sign_name
        signature = self.signatures[sign_name]

        program = self.program.clone()
        paddle_helper.remove_feed_fetch_op(program)
        paddle_helper.set_parameter_trainable(program, trainable)
        paddle_helper.set_op_attr(program, is_test=False)
        self._recovery_parameter(program)
        self._recover_variable_info(program)

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

        return feed_dict, fetch_dict, program

    def parameters(self):
        pass

    def parameter_attrs(self):
        pass

    def default_signature(self):
        return self.default_signature

    def _check_signatures(self):
        assert self.signatures, "signature array should not be None"

        for key, sign in self.signatures.items():
            assert isinstance(sign,
                              Signature), "sign_arr should be list of Signature"

            for input in sign.inputs:
                _tmp_program = input.block.program
                assert self.program == _tmp_program, "all the variable should come from the same program"

            for output in sign.outputs:
                _tmp_program = output.block.program
                assert self.program == _tmp_program, "all the variable should come from the same program"

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
        logger.info("hub version is %s" % version.hub_version)
        logger.info("proto version is %s" % version.proto_version)
        logger.info("paddle version is %s" % paddle.__version__)

        for asset in self.assets:
            pass

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
                if HUB_VAR_PREFIX not in var
            }
            for var, block in varlist.items():
                old_name = var
                new_name = HUB_VAR_PREFIX + old_name
                block._rename_var(old_name, new_name)
            utils.mkdir(self.helper.model_path())
            with open(
                    os.path.join(self.helper.model_path(), "__model__"),
                    "wb") as f:
                f.write(rename_program.desc.serialize_to_string())

            for file in os.listdir(self.helper.model_path()):
                if (file == "__model__" or HUB_VAR_PREFIX in file):
                    continue
                os.rename(
                    os.path.join(self.helper.model_path(), file),
                    os.path.join(self.helper.model_path(),
                                 HUB_VAR_PREFIX + file))

        # Serialize module_desc pb
        module_pb = self.desc.SerializeToString()
        with open(self.helper.module_desc_path(), "wb") as f:
            f.write(module_pb)

        # create processor file
        self._dump_processor()

        # create assets
        self._dump_assets()
