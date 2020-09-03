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

from paddlehub.utils.utils import Version
from paddlehub.utils.io import discard_oe, typein


def get_installed_packages() -> dict:
    '''Get all packages installed in current python environment'''
    return {item.key: Version(item.version) for item in get_installed_distributions()}


def check(package: str, version: str = '') -> bool:
    '''
    Check whether the locally installed python package meets the conditions. If the package is not installed
    locally or the version number does not meet the conditions, return False, otherwise return True.
    '''
    pdict = get_installed_packages()
    if not package in pdict:
        return False
    if not version:
        return True
    return pdict[package].match(version)


def install(package: str, version: str = '', upgrade=False) -> bool:
    '''Install the python package.'''
    with discard_oe():
        cmds = ['install', '{}{}'.format(package, version)]
        if upgrade:
            cmds.append('--upgrade')
        result = pip.main(cmds)
    return result == 0


def uninstall(package: str) -> bool:
    '''Uninstall the python package.'''
    with discard_oe(), typein('y'):
        # type in 'y' to confirm the uninstall operation
        cmds = ['uninstall', '{}'.format(package)]
        result = pip.main(cmds)
    return result == 0
