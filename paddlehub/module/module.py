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
import builtins
import inspect
import importlib
import os
import re
import sys
from typing import Callable, Generic, List, Optional

from easydict import EasyDict

import paddle
from paddlehub.utils import parser, log, utils
from paddlehub.compat import paddle_utils
from paddlehub.compat.module.module_v1 import ModuleV1


class InvalidHubModule(Exception):
    def __init__(self, directory: str):
        self.directory = directory

    def __str__(self):
        return '{} is not a valid HubModule'.format(self.directory)


_module_serving_func = {}
_module_runnable_func = {}


def runnable(func: Callable) -> Callable:
    mod = func.__module__ + '.' + inspect.stack()[1][3]
    _module_runnable_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


def serving(func: Callable) -> Callable:
    mod = func.__module__ + '.' + inspect.stack()[1][3]
    _module_serving_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


class Module(object):
    '''
    '''

    def __new__(cls,
                name: str = None,
                directory: str = None,
                version: str = None,
                source: str = None,
                update: bool = False,
                **kwargs):
        if cls.__name__ == 'Module':
            # This branch come from hub.Module(name='xxx') or hub.Module(directory='xxx')
            if name:
                module = cls.init_with_name(name=name, version=version, source=source, update=update, **kwargs)
            elif directory:
                module = cls.init_with_directory(directory=directory, **kwargs)
        else:
            module = object.__new__(cls)

        return module

    @classmethod
    def load(cls, directory: str) -> Generic:
        '''
        '''
        if directory.endswith(os.sep):
            directory = directory[:-1]

        # If the module description file existed, try to load as ModuleV1
        desc_file = os.path.join(directory, 'module_desc.pb')
        if os.path.exists(desc_file):
            return ModuleV1.load(directory)

        basename = os.path.split(directory)[-1]
        dirname = os.path.join(*list(os.path.split(directory)[:-1]))
        py_module = utils.load_py_module(dirname, '{}.module'.format(basename))

        for _item, _cls in inspect.getmembers(py_module, inspect.isclass):
            _item = py_module.__dict__[_item]
            if hasattr(_item, '_hook_by_hub') and issubclass(_item, RunModule):
                user_module_cls = _item
                break
        else:
            raise InvalidHubModule(directory)

        user_module_cls.directory = directory

        source_info_file = os.path.join(directory, '_source_info.yaml')
        if os.path.exists(source_info_file):
            info = parser.yaml_parser.parse(source_info_file)
            user_module_cls.source = info.get('source', '')
            user_module_cls.branch = info.get('branch', '')
        else:
            user_module_cls.source = ''
            user_module_cls.branch = ''
        return user_module_cls

    @classmethod
    def load_module_info(cls, directory: str) -> EasyDict:
        # If is ModuleV1
        desc_file = os.path.join(directory, 'module_desc.pb')
        if os.path.exists(desc_file):
            return ModuleV1.load_module_info(directory)

        # If is ModuleV2
        module_file = os.path.join(directory, 'module.py')
        with open(module_file, 'r') as file:
            pycode = file.read()
            ast_module = ast.parse(pycode)

            for _body in ast_module.body:
                if not isinstance(_body, ast.ClassDef):
                    continue

                for _decorator in _body.decorator_list:
                    if _decorator.func.id != 'moduleinfo':
                        continue

                    info = {key.arg: key.value.s for key in _decorator.keywords if key.arg != 'meta'}
                    return EasyDict(info)
            else:
                raise InvalidHubModule(directory)

    @classmethod
    def init_with_name(cls,
                       name: str,
                       version: str = None,
                       source: str = None,
                       update: bool = False,
                       branch: str = None,
                       **kwargs):
        '''
        '''
        from paddlehub.module.manager import LocalModuleManager
        manager = LocalModuleManager()
        user_module_cls = manager.search(name, source=source, branch=branch)
        if not user_module_cls or not user_module_cls.version.match(version):
            user_module_cls = manager.install(name=name, version=version, source=source, update=update, branch=branch)

        directory = manager._get_normalized_path(user_module_cls.name)

        # The HubModule in the old version will use the _initialize method to initialize,
        # this function will be obsolete in a future version
        if hasattr(user_module_cls, '_initialize'):
            log.logger.warning(
                'The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object'
            )
            user_module = user_module_cls(directory=directory)
            user_module._initialize(**kwargs)
            return user_module

        if issubclass(user_module_cls, ModuleV1):
            return user_module_cls(directory=directory, **kwargs)

        user_module_cls.directory = directory
        return user_module_cls(**kwargs)

    @classmethod
    def init_with_directory(cls, directory: str, **kwargs):
        '''
        '''
        user_module_cls = cls.load(directory)

        # The HubModule in the old version will use the _initialize method to initialize,
        # this function will be obsolete in a future version
        if hasattr(user_module_cls, '_initialize'):
            log.logger.warning(
                'The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object'
            )
            user_module = user_module_cls(directory=directory)
            user_module._initialize(**kwargs)
            return user_module

        if issubclass(user_module_cls, ModuleV1):
            return user_module_cls(directory=directory, **kwargs)

        user_module_cls.directory = directory
        return user_module_cls(**kwargs)

    @classmethod
    def get_py_requirements(cls):
        '''
        '''
        req_file = os.path.join(cls.directory, 'requirements.txt')
        if not os.path.exists(req_file):
            return []

        with open(req_file, 'r') as file:
            return file.read().split('\n')


class RunModule(object):
    '''
    '''

    def __init__(self, *args, **kwargs):
        # Avoid module being initialized multiple times
        if '_is_initialize' in self.__dict__ and self._is_initialize:
            return

        super(RunModule, self).__init__()
        _run_func_name = self._get_func_name(self.__class__, _module_runnable_func)
        self._run_func = getattr(self, _run_func_name) if _run_func_name else None
        self._serving_func_name = self._get_func_name(self.__class__, _module_serving_func)
        self._is_initialize = True

    def _get_func_name(self, current_cls: Generic, module_func_dict: dict) -> Optional[str]:
        mod = current_cls.__module__ + '.' + current_cls.__name__
        if mod in module_func_dict:
            _func_name = module_func_dict[mod]
            return _func_name
        elif current_cls.__bases__:
            for base_class in current_cls.__bases__:
                return self._get_func_name(base_class, module_func_dict)
        else:
            return None

    # After the 2.0.0rc version, paddle uses the dynamic graph mode by default, which will cause the
    # execution of the static graph model to fail, so compatibility protection is required.
    def __getattribute__(self, attr):
        _attr = object.__getattribute__(self, attr)

        # If the acquired attribute is a built-in property of the object, skip it.
        if re.match('__.*__', attr):
            return _attr
        # If the module is a dygraph model, skip it.
        elif isinstance(self, paddle.nn.Layer):
            return _attr
        # If the acquired attribute is not a class method, skip it.
        elif not inspect.ismethod(_attr):
            return _attr

        return paddle_utils.run_in_static_mode(_attr)

    @classmethod
    def get_py_requirements(cls) -> List[str]:
        '''
        '''
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


def moduleinfo(name: str,
               version: str,
               author: str = None,
               author_email: str = None,
               summary: str = None,
               type: str = None,
               meta=None) -> Callable:
    '''
    '''

    def _wrapper(cls: Generic) -> Generic:
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
            wrap_cls = builtins.type(cls.__name__, _bases, dict(cls.__dict__))

        wrap_cls.name = name
        wrap_cls.version = utils.Version(version)
        wrap_cls.author = author
        wrap_cls.author_email = author_email
        wrap_cls.summary = summary
        wrap_cls.type = type
        wrap_cls._hook_by_hub = True
        return wrap_cls

    return _wrapper
