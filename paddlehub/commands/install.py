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

from paddlehub.common import utils
from paddlehub.module.manager import default_module_manager
from paddlehub.commands.base_command import BaseCommand, ENTRY


class InstallCommand(BaseCommand):
    name = "install"

    def __init__(self, name):
        super(InstallCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Install PaddleHub module."
        self.parser = self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <module_name>' % (ENTRY, name),
            usage='%(prog)s',
            add_help=False)
        #TODO(wuzewu): add --upgrade option

    def execute(self, argv):
        if not argv:
            print("ERROR: Please specify a module name.\n")
            self.help()
            return False
        module_name = argv[0]
        module_version = None if "==" not in module_name else module_name.split(
            "==")[1]
        module_name = module_name if "==" not in module_name else module_name.split(
            "==")[0]
        result, tips, module_dir = default_module_manager.install_module(
            module_name=module_name, module_version=module_version)
        print(tips)
        return True


command = InstallCommand.instance()
