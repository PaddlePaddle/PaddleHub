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
from paddle_hub.commands.base_command import BaseCommand, ENTRY
from paddle_hub.module.manager import default_module_manager
from paddle_hub.module.module import Module
import os
import argparse


class ShowCommand(BaseCommand):
    name = "show"

    def __init__(self, name):
        super(ShowCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Show the specify module's info"
        self.parser = self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <module_name/module_dir>' % (ENTRY, name),
            usage='%(prog)s',
            add_help=False)

    def exec(self, argv):
        if not argv:
            print("ERROR: Please specify a module\n")
            self.help()
            return False

        module_name = argv[0]

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
        #TODO(wuzewu): add more signature info
        print(show_text)
        return True


command = ShowCommand.instance()
