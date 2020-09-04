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

from paddlehub.server import ServerSource, GitSource

PADDLEHUB_PUBLIC_SERVER = 'http://paddlepaddle.org.cn/paddlehub'


class HubServer(object):
    '''PaddleHub server'''

    def __init__(self):
        self.sources = OrderedDict()

    def _generate_source(self, url: str):
        if ServerSource.check(url):
            source = ServerSource(url)
        elif GitSource.check(url):
            source = GitSource(url)
        else:
            raise RuntimeError()
        return source

    def add_source(self, url: str, key: str = None):
        '''Add a module source(GitSource or ServerSource)'''
        key = "source_{}".format(len(self.sources)) if not key else key
        self.sources[key] = self._generate_source(url)

    def remove_source(self, url: str = None, key: str = None):
        '''Remove a module source'''
        self.sources.pop(key)

    def search_module(self, name: str, version: str = None, source: str = None) -> dict:
        '''
        Search PaddleHub module

        Args:
            name(str) : PaddleHub module name
            version(str) : PaddleHub module version
        '''
        return self.search_resouce(type='module', name=name, version=version, source=source)

    def search_resouce(self, type: str, name: str, version: str = None, source: str = None) -> dict:
        '''
        Search PaddleHub Resource

        Args:
            type(str) : Resource type
            name(str) : Resource name
            version(str) : Resource version
        '''
        sources = self.sources.values() if not source else [self._generate_source(source)]
        for source in sources:
            result = source.search_resouce(name=name, type=type, version=version)
            if result:
                return result
        return None


module_server = HubServer()
module_server.add_source(PADDLEHUB_PUBLIC_SERVER)
