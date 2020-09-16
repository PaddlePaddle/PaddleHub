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

import json
import platform
import requests
import sys

import paddlehub
from paddlehub.utils import utils


class ServerConnectionError(Exception):
    def __init__(self, url: str):
        self.url = url

    def __str__(self):
        tips = 'Can\'t connect to Hub Server: {}'.format(self.url)
        return tips


class ServerSource(object):
    '''
    PaddleHub server source

    Args:
        url(str) : Url of the server
        timeout(int) : Request timeout
    '''

    def __init__(self, url: str, timeout: int = 10):
        self._url = url
        self._timeout = timeout

    def search_module(self, name: str, version: str = None) -> dict:
        '''
        Search PaddleHub module

        Args:
            name(str) : PaddleHub module name
            version(str) : PaddleHub module version
        '''
        return self.search_resouce(type='module', name=name, version=version)

    def search_resouce(self, type: str, name: str, version: str = None) -> dict:
        '''
        Search PaddleHub Resource

        Args:
            type(str) : Resource type
            name(str) : Resource name
            version(str) : Resource version
        '''
        payload = {'environments': {}}

        payload['word'] = name
        payload['type'] = type
        if version:
            payload['version'] = version

        # Delay module loading to improve command line speed
        import paddle
        payload['environments']['hub_version'] = paddlehub.__version__
        payload['environments']['paddle_version'] = paddle.__version__
        payload['environments']['python_version'] = '.'.join(map(str, sys.version_info[0:3]))
        payload['environments']['platform_version'] = platform.version()
        payload['environments']['platform_system'] = platform.system()
        payload['environments']['platform_architecture'] = platform.architecture()
        payload['environments']['platform_type'] = platform.platform()

        api = '{}/search'.format(self._url)

        try:
            result = requests.get(api, payload, timeout=self._timeout)
            result = result.json()

            if result['status'] == 0 and len(result['data']) > 0:
                for item in result['data']:
                    if name.lower() == item['name'].lower() and utils.Version(item['version']).match(version):
                        return item
            else:
                print(result)
            return None
        except requests.exceptions.ConnectionError as e:
            raise ServerConnectionError(self._url)

    @classmethod
    def check(cls, url: str) -> bool:
        '''
        Check if the specified url is a valid paddlehub server

        Args:
            url(str) : Url to check
        '''
        try:
            r = requests.get(url + '/search')
            return r.status_code == 200
        except:
            return False
