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

import argparse
import os
from typing import List

from paddlehub.commands import register
from paddlehub.module.manager import LocalModuleManager
from paddlehub.utils import xarfile
from paddlehub.server.server import CacheUpdater


@register(name='hub.install', description='Install PaddleHub module.')
class InstallCommand:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='hub install', add_help=True)
        self.parser.add_argument(
            '--ignore_env_mismatch',
            action='store_true',
            help='Whether to ignore the environment mismatch when installing the Module.')

    def execute(self, argv: List) -> bool:
        if not argv:
            print("ERROR: You must give at least one module to install.")
            return False

        options = [arg for arg in argv if arg.startswith('-')]
        argv = [arg for arg in argv if not arg.startswith('-')]
        args = self.parser.parse_args(options)

        manager = LocalModuleManager()
        for _arg in argv:
            if os.path.exists(_arg) and os.path.isdir(_arg):
                manager.install(directory=_arg)
            elif os.path.exists(_arg) and xarfile.is_xarfile(_arg):
                manager.install(archive=_arg)
            else:
                _arg = _arg.split('==')
                name = _arg[0]
                version = None if len(_arg) == 1 else _arg[1]
                CacheUpdater("hub_install", name, version).start()
                manager.install(name=name, version=version, ignore_env_mismatch=args.ignore_env_mismatch)
        return True
