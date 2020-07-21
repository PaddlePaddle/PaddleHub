# coding:utf-8
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

import optparse
import os
import pip
import sys
from pip._internal.utils.misc import get_installed_distributions
from typing import List, Tuple

from paddlehub.utils.common import Version, generate_tempfile


class PipTool(object):
    '''
    '''

    def get_installed_packages(self) -> List[Tuple[str, str]]:
        '''
        '''
        return {item.key: item.version for item in get_installed_distributions()}

    def check(self, package: str, version: str = '') -> bool:
        '''
        '''
        pdict = self.get_installed_packages()
        if not package in pdict:
            return False
        if not version:
            return True
        return Version(pdict[package]).check(version)

    def install(self, package: str, version: str = '', upgrade=False) -> int:
        '''
        '''
        with generate_tempfile() as file:
            _o = sys.stdout
            _e = sys.stderr
            sys.stdout = sys.stderr = file
            cmds = ['install', '{}{}'.format(package, version)]
            if upgrade:
                cmds.append('--upgrade')
            result = pip.main(cmds)
            sys.stdout = _o
            sys.stderr = _e
        return result

    def uninstall(self, package: str) -> int:
        '''
        '''
        with generate_tempfile() as file:
            _o = sys.stdout
            _e = sys.stderr
            sys.stdout = sys.stderr = file
            cmds = ['uninstall', '{}'.format(package)]

            # typein y to confirm
            with generate_tempfile() as _input:
                _input.write('y\n')
                _i = sys.stdin
                sys.stdin = _input
                result = pip.main(cmds)
                sys.stdin = _i

            sys.stdout = _o
            sys.stderr = _e
        return result
