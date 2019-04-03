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

from paddlehub.common.logger import logger
from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.module.manager import default_module_manager
from paddlehub.module.module import Module
from paddlehub.io.reader import yaml_reader


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

    def exec(self, argv):
        if not argv:
            print("ERROR: Please specify a module or a model\n")
            self.help()
            return False

        module_name = argv[0]

        # nlp model
        model_info = os.path.join(module_name, "info.yml")
        if os.path.exists(model_info):
            model_info = yaml_reader.read(model_info)
            show_text = "Name:%s\n" % model_info['name']
            show_text += "Type:%s\n" % model_info['type']
            show_text += "Version:%s\n" % model_info['version']
            show_text += "Summary:\n"
            show_text += "  %s\n" % model_info['description']
            show_text += "Author:%s\n" % model_info['author']
            show_text += "Author-Email:%s\n" % model_info['author_email']
            print(show_text)
            return True

        cwd = os.getcwd()
        module_dir = default_module_manager.search_module(module_name)
        module_dir = os.path.join(cwd,
                                  module_name) if not module_dir else module_dir
        if not module_dir or not os.path.exists(module_dir):
            return True

        module = Module(module_dir=module_dir)
        show_text = "Name:%s\n" % module.name
        show_text += "Version:%s\n" % module.version
        show_text += "Summary:\n"
        show_text += "  %s\n" % module.summary
        show_text += "Author:%s\n" % module.author
        show_text += "Author-Email:%s\n" % module.author_email
        show_text += "Location:%s\n" % module_dir
        print(show_text)
        return True


command = ShowCommand.instance()
