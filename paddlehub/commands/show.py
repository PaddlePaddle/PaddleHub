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
import argparse

from paddlehub.common import utils
from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.commands.cml_utils import TablePrinter
from paddlehub.module.manager import default_module_manager
from paddlehub.module.module import Module
from paddlehub.io.parser import yaml_parser


class ShowCommand(BaseCommand):
    name = "show"

    def __init__(self, name):
        super(ShowCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Show the information of PaddleHub module."
        self.parser = self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <module_name/module_dir>' % (ENTRY, name),
            usage='%(prog)s',
            add_help=False)

    def show_model_info(self, model_info_file):
        model_info = yaml_parser.parse(model_info_file)
        if utils.is_windows():
            placeholders = [15, 40]
        else:
            placeholders = [15, 50]
        tp = TablePrinter(
            titles=["ModelName", model_info['name']],
            placeholders=placeholders,
            title_colors=["yellow", None],
            title_aligns=["^", "<"])
        tp.add_line(
            contents=["Type", model_info['type']],
            colors=["yellow", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Version", model_info['version']],
            colors=["yellow", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Summary", model_info['description']],
            colors=["yellow", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Author", model_info['author']],
            colors=["yellow", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Author-Email", model_info['author_email']],
            colors=["yellow", None],
            aligns=["^", "<"])
        print(tp.get_text())
        return True

    def show_module_info(self, module_dir):
        module = Module(module_dir=module_dir)
        if utils.is_windows():
            placeholders = [15, 40]
        else:
            placeholders = [15, 50]
        tp = TablePrinter(
            titles=["ModuleName", module.name],
            placeholders=placeholders,
            title_colors=["light_red", None],
            title_aligns=["^", "<"])
        tp.add_line(
            contents=["Version", module.version],
            colors=["light_red", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Summary", module.summary],
            colors=["light_red", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Author", module.author],
            colors=["light_red", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Author-Email", module.author_email],
            colors=["light_red", None],
            aligns=["^", "<"])
        tp.add_line(
            contents=["Location", module_dir[0]],
            colors=["light_red", None],
            aligns=["^", "<"])
        print(tp.get_text())
        return True

    def execute(self, argv):
        if not argv:
            print("ERROR: Please specify a module or a model\n")
            self.help()
            return False

        module_name = argv[0]

        # nlp model
        model_info_file = os.path.join(module_name, "info.yml")
        if os.path.exists(model_info_file):
            self.show_model_info(model_info_file)
            return True

        cwd = os.getcwd()
        module_dir = default_module_manager.search_module(module_name)
        module_dir = (os.path.join(cwd, module_name),
                      None) if not module_dir else module_dir
        if not module_dir or not os.path.exists(module_dir[0]):
            print("%s is not existed!" % module_name)
            return True

        self.show_module_info(module_dir)
        return True


command = ShowCommand.instance()
