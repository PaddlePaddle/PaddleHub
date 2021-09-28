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

import ast
import inspect
import importlib
import os
import sys
from collections import OrderedDict
from typing import List

from paddlehub.module.module import RunModule
from paddlehub.env import SOURCES_HOME
from paddlehub.utils import log, utils


class GitSource(object):
    '''
    Git source for PaddleHub module

    Args:
        url(str) : Url of git repository
        path(str) : Path to store the git repository
    '''

    def __init__(self, url: str, path: str = None):
        # For some environments where git is not installed, we need to set this environment
        # variable to avoid errors.
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        from git import Repo

        self.url = url
        self.path = os.path.join(SOURCES_HOME, utils.md5(url))

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

    def checkout(self, branch: str):
        '''Checkout the current repo to the specified branch.'''
        try:
            self.repo.git.checkout(branch)
            # reload modules
            self.load_hub_modules()
        except:
            utils.record_exception('An error occurred while checkout {}'.format(self.path))

    def update(self):
        '''Update the current repo.'''
        try:
            self.repo.remote().pull(self.repo.branches[0])
            # reload modules
            self.load_hub_modules()
        except:
            self.hub_modules = OrderedDict()
            utils.record_exception('An error occurred while update {}'.format(self.path))

    def load_hub_modules(self):
        if 'hubconf' in sys.modules:
            sys.modules.pop('hubconf')

        sys.path.insert(0, self.path)
        try:
            with open(os.path.join(self.path, 'hubconf.py'), 'r') as file:
                pycode = file.read()
                ast_module = ast.parse(pycode)
                for _body in ast_module.body:
                    if not isinstance(_body, (ast.Import, ast.ImportFrom)):
                        continue

                    if not _body.module.endswith('module'):
                        continue

                    subpath = '.'.join(_body.module.split('.')[:-2])
                    subpath = os.path.join(self.path, subpath)
                    sys.path.insert(0, subpath)

            py_module = importlib.import_module('hubconf')
            for _item, _cls in inspect.getmembers(py_module, inspect.isclass):
                _item = py_module.__dict__[_item]
                if issubclass(_item, RunModule):
                    self.hub_modules[_item.name] = _item
        except:
            self.hub_modules = OrderedDict()
            utils.record_exception('An error occurred while loading {}'.format(self.path))

        sys.path.remove(self.path)

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
        module = self.hub_modules.get(name, None)
        if module and module.version.match(version):
            path = sys.modules[module.__module__].__file__
            path = os.path.dirname(path)
            return [{
                'version': module.version,
                'name': module.name,
                'path': path,
                'class': module.__name__,
                'source': self.url
            }]
        return None

    def get_module_compat_info(self, name: str) -> dict:
        '''Get the version compatibility information of the model.'''
        return {}

    @classmethod
    def check(cls, url: str) -> bool:
        '''
        Check if the specified url is a valid git repository link

        Args:
            url(str) : Url to check
        '''
        import git

        try:
            git.cmd.Git().ls_remote(url)
            return True
        except:
            return False
