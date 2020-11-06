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
import requests
from typing import List

import paddlehub
from paddlehub.utils import platform


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

    def search_module(self, name: str, version: str = None) -> List[dict]:
        '''
        Search PaddleHub module

        Args:
            name(str) : PaddleHub module name
            version(str) : PaddleHub module version
        '''
        return self.search_resource(type='module', name=name, version=version)

    def search_resource(self, type: str, name: str, version: str = None) -> List[dict]:
        '''
        Search PaddleHub Resource

        Args:
            type(str) : Resource type
            name(str) : Resource name
            version(str) : Resource version
        '''
        params = {'environments': platform.get_platform_info()}

        params['word'] = name
        params['type'] = type
        if version:
            params['version'] = version

        # Delay module loading to improve command line speed
        import paddle
        params['hub_version'] = paddlehub.__version__.split('-')[0]
        params['paddle_version'] = paddle.__version__.split('-')[0]

        result = self.request(path='search', params=params)
        if result['status'] == 0 and len(result['data']) > 0:
            return result['data']
        return None

    def get_module_compat_info(self, name: str) -> dict:
        '''Get the version compatibility information of the model.'''

        def _convert_version(version: str) -> List:
            result = []
            # from [1.5.4, 2.0.0] -> 1.5.4,2.0.0
            version = version.replace(' ', '')[1:-1]
            version = version.split(',')
            if version[0] != '-1.0.0':
                result.append('>={}'.format(version[0]))

            if len(version) > 1:
                if version[1] != '99.0.0':
                    result.append('<={}'.format(version[1]))

            return result

        params = {'name': name}
        result = self.request(path='info', params=params)
        if result['status'] == 0 and len(result['data']) > 0:
            infos = {}
            for _info in result['data']['info']:
                infos[_info['version']] = {
                    'url': _info['url'],
                    'paddle_version': _convert_version(_info['paddle_version']),
                    'hub_version': _convert_version(_info['hub_version'])
                }
            return infos

        return {}

    def request(self, path: str, params: dict) -> dict:
        '''Request server.'''
        api = '{}/{}'.format(self._url, path)
        try:
            result = requests.get(api, params, timeout=self._timeout)
            return result.json()
        except requests.exceptions.ConnectionError as e:
            raise ServerConnectionError(self._url)

    def is_connected(self):
        return self.check(self._url)

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
