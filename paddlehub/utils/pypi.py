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

import os
import subprocess
import sys
from typing import IO

from paddlehub.utils.utils import Version
from paddlehub.utils.io import discard_oe


def get_installed_packages() -> dict:
    '''Get all packages installed in current python environment'''
    from pip._internal.utils.misc import get_installed_distributions
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


def install(package: str, version: str = '', upgrade: bool = False, ostream: IO = sys.stdout,
            estream: IO = sys.stderr) -> bool:
    '''Install the python package.'''
    package = package.replace(' ', '')
    if version:
        package = '{}=={}'.format(package, version)

    cmd = '{} -m pip install "{}"'.format(sys.executable, package)

    if upgrade:
        cmd += ' --upgrade'

    result, content = subprocess.getstatusoutput(cmd)
    if result:
        estream.write(content)
    else:
        ostream.write(content)
    return result == 0


def install_from_file(file: str, ostream: IO = sys.stdout, estream: IO = sys.stderr) -> bool:
    '''Install the python package.'''
    cmd = '{} -m pip install -r {}'.format(sys.executable, file)

    result, content = subprocess.getstatusoutput(cmd)
    if result:
        estream.write(content)
    else:
        ostream.write(content)
    return result == 0


def uninstall(package: str, ostream: IO = sys.stdout, estream: IO = sys.stderr) -> bool:
    '''Uninstall the python package.'''
    # type in 'y' to confirm the uninstall operation
    cmd = '{} -m pip uninstall {} -y'.format(sys.executable, package)
    result, content = subprocess.getstatusoutput(cmd)
    if result:
        estream.write(content)
    else:
        ostream.write(content)
    return result == 0
