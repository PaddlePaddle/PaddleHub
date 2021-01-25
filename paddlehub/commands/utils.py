#coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any
from collections import defaultdict


def _CommandDict():
    return defaultdict(_CommandDict)


_commands = _CommandDict()


def register(name: str, description: str = '') -> Any:
    '''
    Register a subcommand in the command list of PaddleHub

    Args:
        name(str) : The name of the command, separated by '.' (e.g, hub.serving)
        description(str) : The description of the specified command showd in the help command, if not description given, this command would not be shown in help command. Default is None.
    '''

    def _warpper(command):
        items = name.split('.')

        com = _commands
        for item in items:
            com = com[item]
        com['_entry'] = command
        if description:
            com['_description'] = description
        return command

    return _warpper


def get_command(name: str) -> Any:
    items = name.split('.')
    com = _commands
    for item in items:
        com = com[item]

    return com['_entry']


def execute():
    '''
    Execute a PaddleHub command and return the status code

    Returns:
         status(int) : Result of the command execution. 0 for a success and 1 for a failure.
    '''
    import sys
    com = _commands
    for idx, _argv in enumerate(['hub'] + sys.argv[1:]):
        if _argv not in com:
            break
        com = com[_argv]
    else:
        idx += 1

    # The method 'execute' of a command instance returns 'True' for a success
    # while 'False' for a failure. Here converts this result into a exit status
    # in bash: 0 for a success and 1 for a failure.
    status = 0 if com['_entry']().execute(sys.argv[idx:]) else 1
    return status
