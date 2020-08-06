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

import base64
import contextlib
import cv2
import math
import os
import sys
import time
import requests
import tempfile
import numpy as np
from typing import Generator
from urllib.parse import urlparse

import packaging.version

import paddlehub.env as hubenv


class Version(packaging.version.Version):
    '''Expand realization of packaging.version.Version'''

    def match(self, condition: str) -> bool:
        '''
        Determine whether the given condition are met

        Args:
            condition(str) : conditions for judgment

        Returns:
            bool: True if the given version condition are met, else False

        Examples:
            from paddlehub.utils import Version

            Version('1.2.0').match('>=1.2.0a')
        '''
        if not condition:
            return True
        if condition.startswith('>='):
            version = condition[2:]
            _comp = self.__ge__
        elif condition.startswith('>'):
            version = condition[1:]
            _comp = self.__gt__
        elif condition.startswith('<='):
            version = condition[2:]
            _comp = self.__le__
        elif condition.startswith('<'):
            version = condition[1:]
            _comp = self.__lt__
        elif condition.startswith('=='):
            version = condition[2:]
            _comp = self.__eq__
        elif condition.startswith('='):
            version = condition[1:]
            _comp = self.__eq__
        else:
            version = condition
            _comp = self.__eq__

        return _comp(Version(version))


class Timer(object):
    ''''''

    def __init__(self, total_step):
        self.total_step = total_step
        self.last_count_step = 0
        self.current_step = 0
        self._is_running = True

    def start(self):
        self.last_time = time.time()
        self.start_time = time.time()

    def stop(self):
        self._is_running = False
        self.end_time = time.time()

    def count(self):
        run_steps = self.current_step - self.last_count_step
        self.last_count_step = self.current_step
        time_used = time.time() - self.last_time
        self.last_time = time.time()
        return time_used / run_steps

    def step(self):
        if not self.current_step >= self.total_step:
            self.current_step += 1
        return self.current_step

    @property
    def is_running(self):
        return self._is_running

    @property
    def eta(self):
        if not self.is_running:
            return '00:00:00'
        scale = self.total_step / self.current_step
        remaining_time = (time.time() - self.start_time) * scale
        return seconds_to_hms(remaining_time)


def seconds_to_hms(seconds):
    ''''''
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = '{}:{}:{}'.format(h, m, s)
    return hms_str


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


@contextlib.contextmanager
def generate_tempfile(directory: str = None):
    '''Generate a temporary file'''
    directory = hubenv.TMP_HOME if not directory else directory
    with tempfile.NamedTemporaryFile(dir=directory) as file:
        yield file


@contextlib.contextmanager
def generate_tempdir(directory: str = None):
    '''Generate a temporary directory'''
    directory = hubenv.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory) as _dir:
        yield _dir


def download(url: str, path: str = None):
    '''
    Download a file

    Args:
        url  (str) : url to be downloaded
        path (str, optional) : path to store downloaded products, default is current work directory

    Examples:
        .. code-block:: python
            from paddlehub.utils.utils import download

            url = 'https://xxxxx.xx/xx.tar.gz'
            download(url, path='./output')
    '''
    for savename, _, _ in download_with_progress(url, path):
        ...
    return savename


def download_with_progress(url: str, path: str = None) -> Generator[str, int, int]:
    '''
    Download a file and return the downloading progress -> Generator[filename, download_size, total_size]

    Args:
        url  (str) : url to be downloaded
        path (str, optional) : path to store downloaded products, default is current work directory

    Examples:
        .. code-block:: python
            from paddlehub.utils.utils import download_with_progress

            url = 'https://xxxxx.xx/xx.tar.gz'
            for filename, download_size, total_szie in download_with_progress(url, path='./output'):
                print(filename, download_size, total_size)
    '''
    path = os.getcwd() if not path else path
    if not os.path.exists(path):
        os.makedirs(path)

    parse_result = urlparse(url)
    savename = parse_result.path.split('/')[-1]
    savename = os.path.join(path, savename)

    res = requests.get(url, stream=True)
    download_size = 0
    total_size = int(res.headers.get('content-length'))
    with open(savename, 'wb') as _file:
        for data in res.iter_content(chunk_size=4096):
            _file.write(data)
            download_size += len(data)
            yield savename, download_size, total_size
