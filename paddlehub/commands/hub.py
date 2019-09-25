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

import six
import sys

from paddlehub.common.logger import logger
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.common import srv_utils
from paddlehub.commands.base_command import BaseCommand
from paddlehub.commands import show
from paddlehub.commands import help
from paddlehub.commands import version
from paddlehub.commands import run
from paddlehub.commands import download


class HubCommand(BaseCommand):
    name = "hub"

    def __init__(self, name):
        super(HubCommand, self).__init__(name)
        self.show_in_help = False

    def execute(self, argv):
        logger.setLevel("NOLOG")

        if not argv:
            help.command.execute(argv)
            exit(1)
            return False
        sub_command = argv[0]
        if not sub_command in BaseCommand.command_dict:
            print("ERROR: unknown command '%s'" % sub_command)
            help.command.execute(argv)
            exit(1)
            return False
        command = BaseCommand.command_dict[sub_command]
        return command.execute(argv[1:])


command = HubCommand.instance()


def main():
    argv = []
    for item in sys.argv:
        if six.PY2:
            argv.append(item.decode(sys_stdin_encoding()).decode("utf8"))
        else:
            argv.append(item)
    command.execute(argv[1:])


if __name__ == "__main__":
    argv = []
    for item in sys.argv:
        if six.PY2:
            argv.append(item.decode(sys_stdin_encoding()).decode("utf8"))
        else:
            argv.append(item)
    command.execute(argv[1:])
