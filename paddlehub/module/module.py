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
import codecs
import inspect
import os
import re
import sys
from typing import Callable
from typing import Generic
from typing import List
from typing import Optional
from typing import Union

import paddle
import paddle2onnx
from easydict import EasyDict

from paddlehub.compat import paddle_utils
from paddlehub.compat.module.module_v1 import ModuleV1
from paddlehub.utils import log
from paddlehub.utils import parser
from paddlehub.utils import utils


class InvalidHubModule(Exception):

    def __init__(self, directory: str):
        self.directory = directory

    def __str__(self):
        return '{} is not a valid HubModule'.format(self.directory)


_module_serving_func = {}
_module_runnable_func = {}


def runnable(func: Callable) -> Callable:
    '''Mark a Module method as runnable, when the command `hub run` is used, the method will be called.'''
    mod = func.__module__ + '.' + inspect.stack()[1][3]
    _module_runnable_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


def serving(func: Callable) -> Callable:
    '''Mark a Module method as serving method, when the command `hub serving` is used, the method will be called.'''
    mod = func.__module__ + '.' + inspect.stack()[1][3]
    _module_serving_func[mod] = func.__name__

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


class RunModule(object):
    '''The base class of PaddleHub Module, users can inherit this class to implement to realize custom class.'''

    def __init__(self, *args, **kwargs):
        super(RunModule, self).__init__()

    def _get_func_name(self, current_cls: Generic, module_func_dict: dict) -> Optional[str]:
        mod = current_cls.__module__ + '.' + current_cls.__name__
        if mod in module_func_dict:
            _func_name = module_func_dict[mod]
            return _func_name
        elif current_cls.__bases__:
            for base_class in current_cls.__bases__:
                base_run_func = self._get_func_name(base_class, module_func_dict)
                if base_run_func:
                    return base_run_func
        else:
            return None

    # After the 2.0.0rc version, paddle uses the dynamic graph mode by default, which will cause the
    # execution of the static graph model to fail, so compatibility protection is required.
    def __getattribute__(self, attr):
        _attr = object.__getattribute__(self, attr)

        # If the acquired attribute is a built-in property of the object, skip it.
        if re.match('__.*__', attr):
            return _attr
        # If the module is a dynamic graph model, skip it.
        elif isinstance(self, paddle.nn.Layer):
            return _attr
        # If the acquired attribute is not a class method, skip it.
        elif not inspect.ismethod(_attr):
            return _attr

        return paddle_utils.run_in_static_mode(_attr)

    @classmethod
    def get_py_requirements(cls) -> List[str]:
        '''Get Module's python package dependency list.'''
        py_module = sys.modules[cls.__module__]
        directory = os.path.dirname(py_module.__file__)
        req_file = os.path.join(directory, 'requirements.txt')
        if not os.path.exists(req_file):
            return []
        with codecs.open(req_file, 'r', encoding='utf8') as file:
            return file.read().split('\n')

    @property
    def _run_func_name(self):
        return self._get_func_name(self.__class__, _module_runnable_func)

    @property
    def _run_func(self):
        return getattr(self, self._run_func_name) if self._run_func_name else None

    @property
    def is_runnable(self) -> bool:
        '''
        Whether the Module is runnable, in other words, whether can we execute the Module through the
        `hub run` command.
        '''
        return True if self._run_func else False

    @property
    def serving_func_name(self):
        return self._get_func_name(self.__class__, _module_serving_func)

    @property
    def _pretrained_model_path(self):
        _pretrained_model_attrs = [
            'pretrained_model_path', 'rec_pretrained_model_path', 'default_pretrained_model_path', 'model_path'
        ]

        for _attr in _pretrained_model_attrs:
            if hasattr(self, _attr):
                path = getattr(self, _attr)
                if os.path.exists(path) and os.path.isfile(path):
                    path = os.path.dirname(path)
                return path

        return None

    def sub_modules(self, recursive: bool = True):
        '''
        Get all sub modules.

        Args:
            recursive(bool): Whether to get sub modules recursively. Default to True.
        '''
        _sub_modules = {}
        for key, item in self.__dict__.items():
            if id(item) == id(self):
                continue

            if isinstance(item, (RunModule, ModuleV1)):
                _sub_modules[key] = item
                if not recursive:
                    continue

                for _k, _v in item.sub_modules(recursive):
                    _sub_modules['{}/{}'.format(key, _k)] = _v

        return _sub_modules

    def save_inference_model(self,
                             dirname: str,
                             model_filename: str = None,
                             params_filename: str = None,
                             input_spec: List[paddle.static.InputSpec] = None,
                             include_sub_modules: bool = True,
                             combined: bool = True):
        '''
        Export the model to Paddle Inference format.

        Args:
            dirname(str): The directory to save the paddle inference model.
            model_filename(str): The name of the saved model file. Default to `__model__`.
            params_filename(str): The name of the saved parameters file, only takes effect when `combined` is True.
                Default to `__params__`.
            input_spec(list): Describes the input of the saved model's forward method, which can be described by
                InputSpec or example Tensor. If None, all input variables of the original Layer's forward method
                would be the inputs of the saved model. Default None.
            include_sub_modules(bool): Whether to export sub modules. Default to True.
            combined(bool): Whether to save all parameters in a combined file. Default to True.
        '''
        if include_sub_modules:
            for key, _sub_module in self.sub_modules().items():
                try:
                    sub_dirname = os.path.normpath(os.path.join(dirname, key))
                    _sub_module.save_inference_model(sub_dirname,
                                                     include_sub_modules=include_sub_modules,
                                                     model_filename=model_filename,
                                                     params_filename=params_filename,
                                                     combined=combined)
                except:
                    utils.record_exception('Failed to save sub module {}'.format(_sub_module.name))

        if isinstance(self, paddle.nn.Layer):
            save_file = os.path.join(dirname, '{}'.format(self.name))
            if not input_spec:
                if hasattr(self, 'input_spec'):
                    input_spec = self.input_spec
                else:
                    _type = self.type.lower()
                    if _type.startswith('cv/image'):
                        input_spec = [paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32')]
                    else:
                        raise RuntimeError(
                            'Module {} lacks `input_spec`, please specify it when calling `save_inference_model`.'.
                            format(self.name))

            net = paddle.jit.to_static(self, input_spec)
            paddle.jit.save(net, save_file)

            log.logger.info('Paddle Inference model saved in {}.'.format(dirname))
            return

        if not self._pretrained_model_path:
            raise RuntimeError('Module {} does not support exporting models in Paddle Inference format.'.format(
                self.name))
        elif not os.path.exists(self._pretrained_model_path):
            log.logger.warning('The model path of Module {} does not exist.'.format(self.name))
            return

        model_filename = '__model__' if not model_filename else model_filename
        if combined:
            params_filename = '__params__' if not params_filename else params_filename

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        _model_filename = None
        _params_filename = None

        if os.path.exists(os.path.join(self._pretrained_model_path, 'model')):
            _model_filename = 'model'

        if os.path.exists(os.path.join(self._pretrained_model_path, 'params')):
            _params_filename = 'params'

        if os.path.exists(os.path.join(self._pretrained_model_path, '__params__')):
            _params_filename = '__params__'

        program, feeded_var_names, target_vars = paddle.static.load_inference_model(
            dirname=self._pretrained_model_path,
            executor=exe,
            model_filename=_model_filename,
            params_filename=_params_filename,
        )

        paddle.static.save_inference_model(dirname=dirname,
                                           main_program=program,
                                           executor=exe,
                                           feeded_var_names=feeded_var_names,
                                           target_vars=target_vars,
                                           model_filename=model_filename,
                                           params_filename=params_filename)

        log.logger.info('Paddle Inference model saved in {}.'.format(dirname))

    def export_onnx_model(self,
                          dirname: str,
                          input_spec: List[paddle.static.InputSpec] = None,
                          include_sub_modules: bool = True,
                          **kwargs):
        '''
        Export the model to ONNX format.

        Args:
            dirname(str): The directory to save the onnx model.
            input_spec(list): Describes the input of the saved model's forward method, which can be described by
                InputSpec or example Tensor. If None, all input variables of the original Layer's forward method
                would be the inputs of the saved model. Default None.
            include_sub_modules(bool): Whether to export sub modules. Default to True.
            **kwargs(dict|optional): Other export configuration options for compatibility, some may be removed in
                the future. Don't use them If not necessary. Refer to https://github.com/PaddlePaddle/paddle2onnx
                for more information.
        '''
        if include_sub_modules:
            for key, _sub_module in self.sub_modules().items():
                try:
                    sub_dirname = os.path.normpath(os.path.join(dirname, key))
                    _sub_module.export_onnx_model(sub_dirname, include_sub_modules=include_sub_modules, **kwargs)
                except:
                    utils.record_exception('Failed to export sub module {}'.format(_sub_module.name))

        if isinstance(self, paddle.nn.Layer):
            save_file = os.path.join(dirname, '{}'.format(self.name))
            if not input_spec:
                if hasattr(self, 'input_spec'):
                    input_spec = self.input_spec
                else:
                    _type = self.type.lower()
                    if _type.startswith('cv/image'):
                        input_spec = [paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32')]
                    else:
                        raise RuntimeError(
                            'Module {} lacks `input_spec`, please specify it when calling `export_onnx_model`.'.format(
                                self.name))

            paddle.onnx.export(self, save_file, input_spec=input_spec, **kwargs)
            return

        if not self._pretrained_model_path:
            raise RuntimeError('Module {} does not support exporting models in ONNX format.'.format(self.name))
        elif not os.path.exists(self._pretrained_model_path):
            log.logger.warning('The model path of Module {} does not exist.'.format(self.name))
            return

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        model_filename = None
        params_filename = None

        if os.path.exists(os.path.join(self._pretrained_model_path, 'model')):
            model_filename = 'model'

        if os.path.exists(os.path.join(self._pretrained_model_path, 'params')):
            params_filename = 'params'

        if os.path.exists(os.path.join(self._pretrained_model_path, '__params__')):
            params_filename = '__params__'

        save_file = os.path.join(dirname, '{}.onnx'.format(self.name))

        program, inputs, outputs = paddle.static.load_inference_model(dirname=self._pretrained_model_path,
                                                                      model_filename=model_filename,
                                                                      params_filename=params_filename,
                                                                      executor=exe)

        paddle2onnx.program2onnx(program=program,
                                 scope=paddle.static.global_scope(),
                                 feed_var_names=inputs,
                                 target_vars=outputs,
                                 save_file=save_file,
                                 **kwargs)


class Module(object):
    '''
    In PaddleHub, Module represents an executable module, which usually a pre-trained model that can be used for end-to-end
    prediction, such as a face detection model or a lexical analysis model, or a pre-trained model that requires finetuning,
    such as BERT/ERNIE. When loading a Module with a specified name, if the Module does not exist locally, PaddleHub will
    automatically request the server or the specified Git source to download the resource.

    Args:
        name(str): Module name.
        directory(str|optional): Directory of the module to be loaded, only takes effect when the `name` is not specified.
        version(str|optional): The version limit of the module, only takes effect when the `name` is specified. When the local
            Module does not meet the specified version conditions, PaddleHub will re-request the server to download the
            appropriate Module. Default to None, This means that the local Module will be used. If the Module does not exist,
            PaddleHub will download the latest version available from the server according to the usage environment.
        source(str|optional): Url of a git repository. If this parameter is specified, PaddleHub will no longer download the
            specified Module from the default server, but will look for it in the specified repository. Default to None.
        update(bool|optional): Whether to update the locally cached git repository, only takes effect when the `source` is
            specified. Default to False.
        branch(str|optional): The branch of the specified git repository. Default to None.
        ignore_env_mismatch(bool|optional): Whether to ignore the environment mismatch when installing the Module. Default to
            False.
    '''

    def __new__(cls,
                *,
                name: str = None,
                directory: str = None,
                version: str = None,
                source: str = None,
                update: bool = False,
                branch: str = None,
                ignore_env_mismatch: bool = False,
                **kwargs):
        if cls.__name__ == 'Module':
            from paddlehub.server.server import CacheUpdater
            # This branch come from hub.Module(name='xxx') or hub.Module(directory='xxx')
            if name:
                module = cls.init_with_name(name=name,
                                            version=version,
                                            source=source,
                                            update=update,
                                            branch=branch,
                                            ignore_env_mismatch=ignore_env_mismatch,
                                            **kwargs)
                CacheUpdater("update_cache", module=name, version=version).start()
            elif directory:
                module = cls.init_with_directory(directory=directory, **kwargs)
                CacheUpdater("update_cache", module=directory, version="0.0.0").start()
        else:
            module = object.__new__(cls)

        return module

    @classmethod
    def load(cls, directory: str) -> Generic:
        '''Load the Module object defined in the specified directory.'''
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

        # In the case of multiple cards, the following code can set each process to use the correct place.
        if issubclass(user_module_cls, paddle.nn.Layer):
            place = paddle.get_device().split(':')[0]
            paddle.set_device(place)

        return user_module_cls

    @classmethod
    def load_module_info(cls, directory: str) -> EasyDict:
        '''Load the infomation of Module object defined in the specified directory.'''
        # If is ModuleV1
        desc_file = os.path.join(directory, 'module_desc.pb')
        if os.path.exists(desc_file):
            return ModuleV1.load_module_info(directory)

        # If is ModuleV2
        module_file = os.path.join(directory, 'module.py')
        with codecs.open(module_file, 'r', encoding='utf8') as file:
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
                       ignore_env_mismatch: bool = False,
                       **kwargs) -> Union[RunModule, ModuleV1]:
        '''Initialize Module according to the specified name.'''
        from paddlehub.module.manager import LocalModuleManager
        manager = LocalModuleManager()
        user_module_cls = manager.search(name, source=source, branch=branch)
        if not user_module_cls or not user_module_cls.version.match(version):
            user_module_cls = manager.install(name=name,
                                              version=version,
                                              source=source,
                                              update=update,
                                              branch=branch,
                                              ignore_env_mismatch=ignore_env_mismatch)

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
    def init_with_directory(cls, directory: str, **kwargs) -> Union[RunModule, ModuleV1]:
        '''Initialize Module according to the specified directory.'''
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


def moduleinfo(name: str,
               version: str,
               author: str = None,
               author_email: str = None,
               summary: str = None,
               type: str = None,
               meta=None) -> Callable:
    '''
    Mark Module information for a python class, and the class will automatically be extended to inherit HubModule. In other words, python classes
    marked with moduleinfo can be loaded through hub.Module.
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
