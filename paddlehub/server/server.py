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

from collections import OrderedDict
from typing import List

import paddlehub.config as hubconf
from paddlehub.server import ServerSource, GitSource
from paddlehub.utils import utils


class HubServer(object):
    '''PaddleHub server'''

    def __init__(self):
        self.sources = OrderedDict()
        self.keysmap = OrderedDict()

    def _generate_source(self, url: str, source_type: str = 'git'):
        if source_type == 'server':
            source = ServerSource(url)
        elif source_type == 'git':
            source = GitSource(url)
        else:
            raise ValueError('Unknown source type {}.'.format(source_type))
        return source

    def _get_source_key(self, url: str):
        return 'source_{}'.format(utils.md5(url))

    def add_source(self, url: str, source_type: str = 'git', key: str = ''):
        '''Add a module source(GitSource or ServerSource)'''
        key = self._get_source_key(url) if not key else key
        self.keysmap[url] = key
        self.sources[key] = self._generate_source(url, source_type)

    def remove_source(self, url: str = None, key: str = None):
        '''Remove a module source'''
        self.sources.pop(key)

    def get_source(self, url: str):
        '''Get a module source by url'''
        key = self.keysmap.get(url)
        if not key:
            return None
        return self.sources.get(key)

    def get_source_by_key(self, key: str):
        '''Get a module source by key'''
        return self.sources.get(key)

    def search_module(self,
                      name: str,
                      version: str = None,
                      source: str = None,
                      update: bool = False,
                      branch: str = None) -> List[dict]:
        '''
        Search PaddleHub module

        Args:
            name(str) : PaddleHub module name
            version(str) : PaddleHub module version
        '''
        return self.search_resource(
            type='module', name=name, version=version, source=source, update=update, branch=branch)

    def search_resource(self,
                        type: str,
                        name: str,
                        version: str = None,
                        source: str = None,
                        update: bool = False,
                        branch: str = None) -> List[dict]:
        '''
        Search PaddleHub Resource

        Args:
            type(str) : Resource type
            name(str) : Resource name
            version(str) : Resource version
        '''
        sources = self.sources.values() if not source else [self._generate_source(source)]
        for source in sources:
            if isinstance(source, GitSource) and update:
                source.update()

            if isinstance(source, GitSource) and branch:
                source.checkout(branch)

            result = source.search_resource(name=name, type=type, version=version)
            if result:
                return result
        return []

    def get_module_compat_info(self, name: str, source: str = None) -> dict:
        '''Get the version compatibility information of the model.'''
        sources = self.sources.values() if not source else [self._generate_source(source)]
        for source in sources:
            result = source.get_module_compat_info(name=name)
            if result:
                return result
        return {}


module_server = HubServer()
module_server.add_source(hubconf.server, source_type='server', key='default_hub_server')
