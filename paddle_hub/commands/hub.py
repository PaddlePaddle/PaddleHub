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
from paddle_hub.commands import show
from paddle_hub.commands import help
from paddle_hub.commands import version
from paddle_hub.commands import run
from paddle_hub.commands import download
import sys


class HubCommand(BaseCommand):
    name = "hub"

    def __init__(self, name):
        super(HubCommand, self).__init__(name)
        self.show_in_help = False
        # yapf: disable
        self.add_arg('command',  str, None,  "command to run" )
        # yapf: enable

    def exec(self, argv):
        args = self.parser.parse_args(argv[1:2])

        command_dict = {
            'show': show.command.exec,
            'run': run.command.exec,
            'download': download.command.exec,
            'help': help.command.exec,
            'version': version.command.exec
        }

        if not args.command in command_dict:
            logger.critical("command %s not supported!" % args.command)
            exit(1)

        command = command_dict[args.command]
        command(argv[2:])


command = HubCommand.instance()

if __name__ == "__main__":
    command.exec(sys.argv)
