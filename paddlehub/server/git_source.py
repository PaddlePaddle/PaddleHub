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

import inspect
import importlib
import os
import sys
from collections import OrderedDict
from urllib.parse import urlparse

import git
from git import Repo

from paddlehub.module.module import Module as HubModule
from paddlehub.env import SOURCES_HOME
from paddlehub.utils import log


class GitSource(object):
    def __init__(self, url, path=None):
        self.url = url
        self._parse_result = urlparse(self.url)

        self.path = os.path.join(SOURCES_HOME, self._parse_result.path[1:])
        if self.path.endswith('.git'):
            self.path = self.path[:-4]

        if not os.path.exists(self.path):
            log.logger.info('Git repository {} does not exist, download from remote.'.format(self.url))
            self.repo = Repo.clone_from(self.url, self.path)
        else:
            log.logger.info('Git repository {} is located at {}.'.format(self.url, self.path))
            self.repo = Repo(self.path)

        self.hub_modules = OrderedDict()
        self.load_hub_modules()

    def update(self):
        self.repo.remote().pull()

    def load_hub_modules(self):
        if 'hubconf' in sys.modules:
            sys.modules.pop('hubconf')

        sys.path.insert(0, self.path)
        try:
            py_module = importlib.import_module('hubconf')
            for _item, _cls in inspect.getmembers(py_module, inspect.isclass):
                _item = py_module.__dict__[_item]
                if issubclass(_item, HubModule):
                    self.hub_modules[_item.name] = _item
        except:
            raise
            log.logger.warning('An error occurred while loading {}'.format(self.path))
        sys.path.remove(self.path)

    def search_module(self, name, version=None):
        return self.search_resouce(type='module', name=name, version=version)

    def search_resouce(self, type, name, version=None):
        module = self.hub_modules.get(name, None)
        if module and module.version.match(version):
            return {
                'version': module.version,
                'name': module.name,
                'path': self.path,
                'class': module.__name__,
                'source': self.url
            }
        return None

    @classmethod
    def check(cls, url):
        try:
            git.cmd.Git().ls_remote(url)
            return True
        except:
            return False
