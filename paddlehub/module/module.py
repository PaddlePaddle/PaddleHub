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

import paddle.fluid as fluid

from paddlehub.utils import utils


class InvalidHubModule(Exception):
    def __init__(self, directory):
        self.directory = directory

    def __str__(self):
        return '{} is not a valid HubModule'.format(self.directory)


_module_serving_func = {}
_module_runnable_func = {}


def runnable(func):
    mod = func.__module__ + '.' + inspect.stack()[1][3]
    _module_runnable_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


def serving(func):
    mod = func.__module__ + '.' + inspect.stack()[1][3]
    _module_serving_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


class Module(object):
    def __new__(cls, name: str = None, directory: str = None, version: str = None, **kwargs):
        if cls.__name__ == 'Module':
            if name:
                module = cls.init_with_name(name=name, version=version, **kwargs)
            elif directory:
                module = cls.init_with_directory(directory=directory, **kwargs)
        else:
            raise RuntimeError()

        module.directory = directory
        return module

    @classmethod
    def load(cls, directory: str):
        if directory.endswith(os.sep):
            directory = directory[:-1]

        basename = os.path.split(directory)[-1]
        dirname = os.path.join(*list(os.path.split(directory)[:-1]))

        sys.path.insert(0, dirname)
        py_module = importlib.import_module('{}.module'.format(basename))

        for _item, _cls in inspect.getmembers(py_module, inspect.isclass):
            _item = py_module.__dict__[_item]
            if hasattr(_item, '_hook_by_hub') and issubclass(_item, RunModule):
                user_module_cls = _item
                break
        else:
            raise InvalidHubModule(directory)
        sys.path.pop(0)

        return user_module_cls

    @classmethod
    def init_with_name(cls, name: str, version: str = None, **kwargs):
        from paddlehub.module.manager import LocalModuleManager
        manager = LocalModuleManager()
        search_result = manager.search(name)
        user_module_cls = search_result.get('module', None)
        directory = search_result.get('directory', None)
        if not user_module_cls or not user_module_cls.version.match(version):
            user_module_cls = manager.install(name, version)

        return user_module_cls(**kwargs)

    @classmethod
    def init_with_directory(cls, directory: str, **kwargs):
        user_module_cls = cls.load(directory)
        return user_module_cls(**kwargs)


class RunModule(object):
    def __init__(self, *args, **kwargs):
        # Avoid module being initialized multiple times
        if '_is_initialize' in self.__dict__ and self._is_initialize:
            return

        super(RunModule, self).__init__()
        _run_func_name = self._get_func_name(self.__class__, _module_runnable_func)
        self._run_func = getattr(self, _run_func_name) if _run_func_name else None
        self._serving_func_name = self._get_func_name(self.__class__, _module_serving_func)
        self._is_initialize = True

    def _get_func_name(self, current_cls, module_func_dict):
        mod = current_cls.__module__ + '.' + current_cls.__name__
        if mod in module_func_dict:
            _func_name = module_func_dict[mod]
            return _func_name
        elif current_cls.__bases__:
            for base_class in current_cls.__bases__:
                return self._get_func_name(base_class, module_func_dict)
        else:
            return None

    @classmethod
    def get_py_requirements(cls):
        py_module = sys.modules[cls.__module__]
        directory = os.path.dirname(py_module.__file__)
        req_file = os.path.join(directory, 'requirements.txt')
        if not os.path.exists(req_file):
            return []
        with open(req_file, 'r') as file:
            return file.read()

    @property
    def is_runnable(self) -> bool:
        return self._run_func != None


sys_type = type


def moduleinfo(name: str,
               version: str,
               author: str = None,
               author_email: str = None,
               summary: str = None,
               type: str = None,
               meta=None):
    def _wrapper(cls):
        wrap_cls = cls
        _meta = RunModule if not meta else meta
        if not issubclass(cls, _meta):
            _bases = []
            for _b in cls.__bases__:
                if issubclass(_meta, _b):
                    continue
                _bases.append(_b)
            _bases.append(_meta)
            _bases = tuple(_bases)
            wrap_cls = sys_type(cls.__name__, _bases, dict(cls.__dict__))

        wrap_cls.name = name
        wrap_cls.version = utils.Version(version)
        wrap_cls.author = author
        wrap_cls.author_email = author_email
        wrap_cls.summary = summary
        wrap_cls.type = type
        wrap_cls._hook_by_hub = True
        return wrap_cls

    return _wrapper
