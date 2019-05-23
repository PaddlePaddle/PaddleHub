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

from paddlehub.commands.base_command import BaseCommand


class HelpCommand(BaseCommand):
    name = "help"

    def __init__(self, name):
        super(HelpCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Show help for commands."

    def get_all_commands(self):
        return BaseCommand.command_dict

    def execute(self, argv):
        hub_command = BaseCommand.command_dict["hub"]
        help_text = "\n"
        help_text += "Usage:\n"
        help_text += "%s <command> [options]\n" % hub_command.name
        help_text += "\n"
        help_text += "Commands:\n"
        for command_name, command in self.get_all_commands().items():
            if not command.show_in_help or not command.description:
                continue
            help_text += "  %-15s\t\t%s\n" % (command.name, command.description)

        print(help_text)
        return True


command = HelpCommand.instance()
