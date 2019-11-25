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

import argparse
import json
import os
import sys
import ast

import six
import pandas
import imghdr
import cv2
import numpy as np

from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.io.parser import yaml_parser, txt_parser
from paddlehub.module.manager import default_module_manager
from paddlehub.common import utils
from paddlehub.common.arg_helper import add_argument, print_arguments
import paddlehub as hub


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


class RunCommand(BaseCommand):
    name = "run"

    def __init__(self, name):
        super(RunCommand, self).__init__(name)
        self.show_in_help = True
        self.name = name
        self.description = "Run the specific module."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <module>' % (ENTRY, self.name),
            usage='%(prog)s',
            add_help=False)
        self.module = None

    def find_module(self, module_name):
        module_dir = default_module_manager.search_module(module_name)
        if not module_dir:
            if os.path.exists(module_name):
                module_dir = (module_name, None)
            else:
                print("Install Module %s" % module_name)
                extra = {"command": "install"}
                result, tips, module_dir = default_module_manager.install_module(
                    module_name, extra=extra)
                print(tips)
                if not result:
                    return None

        return hub.Module(module_dir=module_dir)

    def add_module_config_arg(self):
        configs = self.module.processor.configs()
        for config in configs:
            if not config["dest"].startswith("--"):
                config["dest"] = "--%s" % config["dest"]
            self.arg_config_group.add_argument(
                config["dest"],
                type=config['type'],
                default=config['default'],
                help=config['help'])

        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU for prediction")

        self.arg_config_group.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help="batch size for prediction")

        self.arg_config_group.add_argument(
            '--config',
            type=str,
            default=None,
            help="config file in yaml format")

    def add_module_input_arg(self):
        module_type = self.module.type.lower()
        expect_data_format = self.module.processor.data_format(
            self.module.default_signature.name)
        self.arg_input_group.add_argument(
            '--input_file',
            type=str,
            default=None,
            help="file contain input data")
        self.arg_input_group.add_argument(
            '--use_strip',
            type=ast.literal_eval,
            default=True,
            help=
            "whether need to strip whitespace characters from the beginning and the end of the line in the file or not."
        )
        if len(expect_data_format) == 1:
            if module_type.startswith("cv"):
                self.arg_input_group.add_argument(
                    '--input_path',
                    type=str,
                    default=None,
                    help="path of image/video to predict")
            elif module_type.startswith("nlp"):
                self.arg_input_group.add_argument(
                    '--input_text',
                    type=str,
                    default=None,
                    help="text to predict")
        else:
            for key in expect_data_format.keys():
                help_str = None
                if 'help' in expect_data_format[key]:
                    help_str = expect_data_format[key]['help']
                self.arg_input_group.add_argument(
                    "--%s" % key, type=str, default=None, help=help_str)

    def get_config(self):
        yaml_config = {}
        if self.args.config:
            yaml_config = yaml_parser.parse(self.args.config)
        module_config = yaml_config.get("config", {})
        for _config in self.module.processor.configs():
            key = _config['dest']
            module_config[key] = self.args.__dict__[key]
        return module_config

    def get_data(self):
        module_type = self.module.type.lower()
        expect_data_format = self.module.processor.data_format(
            self.module.default_signature.name)
        input_data = {}
        if len(expect_data_format) == 1:
            key = list(expect_data_format.keys())[0]
            if self.args.input_file:
                input_data[key] = txt_parser.parse(self.args.input_file,
                                                   self.args.use_strip)
            else:
                if module_type.startswith("cv"):
                    if hasattr(self.args, "input_path"):
                        self.check_file()
                    input_data[key] = [self.args.input_path]
                elif module_type.startswith("nlp"):
                    input_data[key] = [self.args.input_text]
        else:
            for key in expect_data_format.keys():
                input_data[key] = [self.args.__dict__[key]]

            if self.args.input_file:
                input_data = pandas.read_csv(self.args.input_file, sep="\t")

        return input_data

    def check_data(self, data):
        expect_data_format = self.module.processor.data_format(
            self.module.default_signature.name)

        if len(data.keys()) != len(expect_data_format.keys()):
            print(
                "ERROR: The number of keys in input file is inconsistent with expectations."
            )
            raise DataFormatError

        if isinstance(data, pandas.DataFrame):
            if data.isnull().sum().sum() != 0:
                print(
                    "ERROR: The number of values in input file is inconsistent with expectations."
                )
                raise DataFormatError

        for key, values in data.items():

            if not key in expect_data_format.keys():
                print("ERROR! Key <%s> in input file is unexpected.\n" % key)
                raise DataFormatError

            for value in values:
                if not value:
                    print(
                        "ERROR: The number of values in input file is inconsistent with expectations."
                    )
                    raise DataFormatError

    def check_file(self):
        file_path = self.args.input_path
        if not os.path.exists(file_path):
            raise RuntimeError("ERROR: File %s is not exist." % file_path)
        if imghdr.what(file_path) is not None or \
                cv2.VideoCapture(file_path).get(cv2.CAP_PROP_FRAME_COUNT) > 1:
            return

        raise RuntimeError("ERROR: Format of %s is illegal." % file_path)

    def execute(self, argv):
        if not argv:
            print("ERROR: Please specify a module name.\n")
            self.help()
            return False

        module_name = argv[0]

        self.parser.prog = '%s %s %s' % (ENTRY, self.name, module_name)
        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Data input to the module")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required")

        self.module = self.find_module(module_name)
        if not self.module:
            return False

        # If the module is not executable, give an alarm and exit
        if not self.module.default_signature:
            print("ERROR! Module %s is not executable." % module_name)
            return False

        self.module.check_processor()
        self.add_module_config_arg()
        self.add_module_input_arg()

        if not argv[1:]:
            self.help()
            return False

        self.args = self.parser.parse_args(argv[1:])

        config = self.get_config()
        data = self.get_data()

        try:
            self.check_data(data)
        except DataFormatError:
            self.help()
            return False

        results = self.module(
            sign_name=self.module.default_signature.name,
            data=data,
            use_gpu=self.args.use_gpu,
            batch_size=self.args.batch_size,
            **config)

        if six.PY2:
            try:
                results = json.dumps(
                    results, encoding="utf8", ensure_ascii=False)
            except:
                pass

        print(results)

        return True


command = RunCommand.instance()
