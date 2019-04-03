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

from paddle_hub.common.logger import logger
from paddle_hub.commands.base_command import BaseCommand, ENTRY
from paddle_hub.io.reader import csv_reader, yaml_reader
from paddle_hub.module.manager import default_module_manager
from paddle_hub.common import utils
from paddle_hub.common.arg_helper import add_argument, print_arguments

import paddle_hub as hub
import argparse
import os


class RunCommand(BaseCommand):
    name = "run"

    def __init__(self, name):
        super(RunCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Run the specific module."
        self.parser = self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <module>' % (ENTRY, name),
            usage='%(prog)s',
            add_help=False)

    def parse_args_with_module(self, module, argv):
        module_type = module.type.lower()
        # yapf: disable
        if module_type.startswith("cv"):
            self.add_arg('--config',        str, None,  "config file in yaml format" )
            self.add_arg('--signature',     str, None,  "signature to run" )
            self.add_arg('--input_path',    str, None,  "path of image to predict" )
            self.add_arg('--input_file',    str, None,  "file contain paths of images" )
            self.args = self.parser.parse_args(argv)
            self.args.data = self.args.input_path
            self.args.dataset = self.args.input_file
        elif module_type.startswith("nlp"):
            self.add_arg('--config',        str, None,  "config file in yaml format" )
            self.add_arg('--signature',     str, None,  "signature to run" )
            self.add_arg('--input_text',    str, None,  "text to predict" )
            self.add_arg('--input_file',    str, None,  "file contain texts" )
            self.args = self.parser.parse_args(argv)
            self.args.data = self.args.input_text
            self.args.dataset = self.args.input_file
        # yapf: enable

    def demo_with_module(self, module):
        module_type = module.type.lower()
        entry = hub.commands.base_command.ENTRY
        if module_type.startswith("cv"):
            demo = "%s %s %s --input_path <IMAGE_PATH>" % (entry, self.name,
                                                           module.name)
        elif module_type.startswith("nlp"):
            demo = "%s %s %s --input_text \"TEXT_TO_PREDICT\"" % (
                entry, self.name, module.name)
        else:
            demo = "%s %s %s" % (entry, self.name, module.name)
        return demo

    def exec(self, argv):
        if not argv:
            print("ERROR: Please specify a module name.\n")
            self.help()
            return False
        module_name = argv[0]

        module_dir = default_module_manager.search_module(module_name)
        if not module_dir:
            if os.path.exists(module_name):
                module_dir = module_name
            else:
                print("Install Module %s" % module_name)
                result, tips, module_dir = default_module_manager.install_module(
                    module_name)
                print(tips)
                if not result:
                    return False

        module = hub.Module(module_dir=module_dir)
        self.parse_args_with_module(module, argv[1:])

        if not module.default_signature:
            print("ERROR! Module %s is not callable." % module_name)
            return False

        if not self.args.signature:
            self.args.signature = module.default_signature.name
        # module processor check
        module.check_processor()
        expect_data_format = module.processor.data_format(self.args.signature)

        # get data dict
        if self.args.data:
            input_data_key = list(expect_data_format.keys())[0]
            origin_data = {input_data_key: [self.args.data]}
        elif self.args.dataset:
            origin_data = csv_reader.read(self.args.dataset)
        else:
            print("ERROR! Please specify data to predict.\n")
            print("Summary:\n    %s\n" % module.summary)
            print("Example:\n    %s" % self.demo_with_module(module))
            return False

        # data_format check
        if not self.args.config:
            assert len(expect_data_format) == 1
            origin_data_key = list(origin_data.keys())[0]
            input_data_key = list(expect_data_format.keys())[0]
            input_data = {input_data_key: origin_data[origin_data_key]}
            config = {}
        else:
            yaml_config = yaml_reader.read(self.args.config)
            if len(expect_data_format) == 1:
                origin_data_key = list(origin_data.keys())[0]
                input_data_key = list(expect_data_format.keys())[0]
                input_data = {input_data_key: origin_data[origin_data_key]}
            else:
                input_data_format = yaml_config['input_data']
                assert len(input_data_format) == len(expect_data_format)
                for key, value in expect_data_format.items():
                    assert key in input_data_format
                    assert value['type'] == hub.DataType.type(
                        input_data_format[key]['type'])

                input_data = {}
                for key, value in yaml_config['input_data'].items():
                    input_data[key] = origin_data[value['key']]
            config = yaml_config.get("config", {})
        # run module with data
        print(module(sign_name=self.args.signature, data=input_data, **config))


command = RunCommand.instance()
