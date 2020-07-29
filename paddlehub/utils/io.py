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

import contextlib
import sys

from paddlehub.utils.utils import generate_tempfile


@contextlib.contextmanager
def redirect_istream(stream):
    '''reset the standard input stream to the specified stream'''
    _t = sys.stdin
    sys.stdin = stream
    yield
    sys.stdin = _t


@contextlib.contextmanager
def redirect_ostream(stream):
    '''reset the standard output stream to the specified stream'''
    _t = sys.stdout
    sys.stdout = stream
    yield
    sys.stdout = _t


@contextlib.contextmanager
def redirect_estream(stream):
    '''reset the standard error stream to the specified stream'''
    _t = sys.stderr
    sys.stderr = stream
    yield
    sys.stderr = _t


@contextlib.contextmanager
def discard_oe():
    with generate_tempfile() as _file:
        with open(_file.name, 'w') as _stream:
            with redirect_ostream(_stream):
                with redirect_estream(_stream):
                    yield


@contextlib.contextmanager
def confirm(char='y'):
    # typein char to confirm
    with generate_tempfile() as _file:
        with open(_file.name, 'w+') as _stream:
            with redirect_istream(_stream):
                _stream.write('{}\n'.format(char))
                _stream.seek(0)
                yield
