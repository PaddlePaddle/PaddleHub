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
import os
import requests
import threading
import time
import yaml

from collections import OrderedDict
from typing import List

import paddle
import paddlehub
import paddlehub.config as hubconf
from paddlehub.config import cache_config
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


def uri_path(server_url, api):
    srv = server_url
    if server_url.endswith('/'):
        srv = server_url[:-1]
    if api.startswith('/'):
        srv += api
    else:
        api = '/' + api
        srv += api
    return srv


def hub_request(api, params, extra=None, timeout=8):
    params['hub_version'] = paddlehub.__version__.split('-')[0]
    params['paddle_version'] = paddle.__version__.split('-')[0]

    params["extra"] = json.dumps(extra)
    r = requests.get(api, params, timeout=timeout)
    return r.json()


class CacheUpdater(threading.Thread):
    def __init__(self, command="update_cache", module=None, version=None, addition=None):
        threading.Thread.__init__(self)
        self.command = command
        self.module = module
        self.version = version
        self.addition = addition

    def update_resource_list_file(self, command="update_cache", module=None, version=None, addition=None):
        payload = {'word': module}
        if version:
            payload['version'] = version
        api_url = uri_path(hubconf.server, 'search')
        cache_path = os.path.join("~")
        hub_name = cache_config.hub_name
        if os.path.exists(cache_path):
            extra = {"command": command, "mtime": os.stat(cache_path).st_mtime, "hub_name": hub_name}
        else:
            extra = {
                "command": command,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "hub_name": hub_name
            }
        if addition is not None:
            extra.update({"addition": addition})
        try:
            r = hub_request(api_url, payload, extra, timeout=1)
            if r.get("update_cache", 0) == 1:
                with open(cache_path, 'w+') as fp:
                    yaml.safe_dump({'resource_list': r['data']}, fp)
        except Exception as err:
            pass

    def run(self):
        self.update_resource_list_file(self.command, self.module, self.version, self.addition)


module_server = HubServer()
module_server.add_source(hubconf.server, source_type='server', key='default_hub_server')
