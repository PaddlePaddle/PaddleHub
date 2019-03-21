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
from paddle_hub.tools.logger import logger
from paddle_hub.commands.base_command import BaseCommand
import paddle_hub as hub
from paddle_hub.io.reader import csv_reader, yaml_reader
from paddle_hub.tools import utils
from paddle_hub.tools.arg_helper import add_argument, print_arguments


class RunCommand(BaseCommand):
    name = "run"

    def __init__(self, name):
        super(RunCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Run the specify module"
        # yapf: disable
        self.add_arg('--config',    str, None,  "config file in yaml format" )
        self.add_arg('--dataset',   str, None,  "dataset be used" )
        self.add_arg('--module',    str, None,  "module to run" )
        self.add_arg('--signature', str, None,  "signature to run" )
        # yapf: enable

    def _check_module(self):
        if not self.args.module:
            logger.critical("lack of module")
            self.help()
            exit(1)

    def _check_dataset(self):
        if not self.args.dataset:
            logger.critical("lack of dataset file")
            self.help()
            exit(1)
        if not utils.is_csv_file(self.args.dataset):
            logger.critical("dataset file should in csv format")
            self.help()
            exit(1)

    def _check_config(self):
        if not self.args.config:
            logger.critical("lack of config file")
            self.help()
            exit(1)
        if not utils.is_yaml_file(self.args.config):
            logger.critical("config file should in yaml format")
            self.help()
            exit(1)

    def help(self):
        self.parser.print_help()

    def exec(self, argv):
        self.args = self.parser.parse_args(argv)
        self.print_args()

        self._check_module()
        self._check_dataset()
        self._check_config()

        module = hub.Module(module_dir=self.args.module)
        yaml_config = yaml_reader.read(self.args.config)

        if not self.args.signature:
            self.args.signature = module.default_signature().name

        # module processor check
        module.check_processor()
        # data_format check
        expect_data_format = module.processor.data_format(self.args.signature)
        input_data_format = yaml_config['input_data']
        assert len(input_data_format) == len(expect_data_format)
        for key, value in expect_data_format.items():
            assert key in input_data_format
            assert value['type'] == hub.DataType.type(
                input_data_format[key]['type'])

        # get data dict
        origin_data = csv_reader.read(self.args.dataset)
        input_data = {}
        for key, value in yaml_config['input_data'].items():
            type_reader = hub.DataType.type_reader(value['type'])
            input_data[key] = list(
                map(type_reader.read, origin_data[value['key']]))

        # run module with data
        print(
            module(
                sign_name=self.args.signature,
                data=input_data,
                **yaml_config['config']))


command = RunCommand.instance()
