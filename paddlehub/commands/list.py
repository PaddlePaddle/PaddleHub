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

from paddlehub.common import utils
from paddlehub.common.downloader import default_downloader
from paddlehub.module.manager import default_module_manager
from paddlehub.commands.base_command import BaseCommand
from paddlehub.commands.cml_utils import TablePrinter


class ListCommand(BaseCommand):
    name = "list"

    def __init__(self, name):
        super(ListCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "List all installed PaddleHub modules."

    def execute(self, argv):
        all_modules = default_module_manager.all_modules()
        if utils.is_windows():
            placeholders = [20, 40]
        else:
            placeholders = [25, 50]
        tp = TablePrinter(
            titles=["ModuleName", "Path"], placeholders=placeholders)
        for module_name, module_dir in all_modules.items():
            tp.add_line(contents=[module_name, module_dir[0]])
        print(tp.get_text())
        return True


command = ListCommand.instance()
