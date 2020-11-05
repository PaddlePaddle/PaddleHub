# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict
from typing import Callable


class TaskHooks(object):
    '''TaskHooks can handle some tasks during the spectific event.'''

    def __init__(self):
        self._registered_hooks = {
            'build_env_start_event': OrderedDict(),
            'build_env_end_event': OrderedDict(),
            'finetune_start_event': OrderedDict(),
            'finetune_end_event': OrderedDict(),
            'predict_start_event': OrderedDict(),
            'predict_end_event': OrderedDict(),
            'eval_start_event': OrderedDict(),
            'eval_end_event': OrderedDict(),
            'log_interval_event': OrderedDict(),
            'save_ckpt_interval_event': OrderedDict(),
            'eval_interval_event': OrderedDict(),
            'run_step_event': OrderedDict(),
        }
        self._hook_params_num = {
            'build_env_start_event': 1,
            'build_env_end_event': 1,
            'finetune_start_event': 1,
            'finetune_end_event': 2,
            'predict_start_event': 1,
            'predict_end_event': 2,
            'eval_start_event': 1,
            'eval_end_event': 2,
            'log_interval_event': 2,
            'save_ckpt_interval_event': 1,
            'eval_interval_event': 1,
            'run_step_event': 2,
        }

    def add(self, hook_type: str, name: str = None, func: Callable = None):
        '''
        add the handler function to spectific event.
        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name, default None
            func (func): the handler function, default None
        '''
        if not func or not callable(func):
            raise TypeError('The hook function is empty or it is not a function')
        if name == None:
            name = 'hook_%s' % id(func)

        # check validity
        if not isinstance(name, str) or name.strip() == '':
            raise TypeError('The hook name must be a non-empty string')
        if hook_type not in self._registered_hooks:
            raise ValueError('hook_type: %s does not exist' % (hook_type))
        if name in self._registered_hooks[hook_type]:
            raise ValueError('name: %s has existed in hook_type:%s, use modify method to modify it' % (name, hook_type))
        else:
            args_num = len(inspect.getfullargspec(func).args)
            if args_num != self._hook_params_num[hook_type]:
                raise ValueError('The number of parameters to the hook hook_type:%s should be %i' %
                                 (hook_type, self._hook_params_num[hook_type]))
            self._registered_hooks[hook_type][name] = func

    def delete(self, hook_type: str, name: str):
        '''
        delete the handler function of spectific event.
        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
        '''
        if self.exist(hook_type, name):
            del self._registered_hooks[hook_type][name]
        else:
            raise ValueError(
                'No hook_type: %s exists or name: %s does not exist in hook_type: %s' % (hook_type, name, hook_type))

    def modify(self, hook_type: str, name: str, func: Callable):
        '''
        modify the handler function of spectific event.
        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
            func (func): the new handler function
        '''
        if not (isinstance(name, str) and callable(func)):
            raise TypeError('The hook name must be a string, and the hook function must be a function')
        if self.exist(hook_type, name):
            self._registered_hooks[hook_type][name] = func
        else:
            raise ValueError(
                'No hook_type: %s exists or name: %s does not exist in hook_type: %s' % (hook_type, name, hook_type))

    def exist(self, hook_type: str, name: str) -> bool:
        '''
        check if the the handler function of spectific event is existing.
        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
        Returns:
            bool: True or False
        '''
        if hook_type not in self._registered_hooks \
                or name not in self._registered_hooks[hook_type]:
            return False
        else:
            return True

    def info(self, show_default: bool = False) -> str:
        '''
        get the hooks information, including the source code.
        Args:
            show_default (bool): show the information of Paddlehub default hooks or not, default False
        Returns:
            str: the formatted string of the hooks information
        '''
        # formatted output the source code
        ret = ''
        for hook_type, hooks in self._registered_hooks.items():
            already_print_type = False
            for name, func in hooks.items():
                if name == 'default' and not show_default:
                    continue
                if not already_print_type:
                    ret += 'hook_type: %s{\n' % hook_type
                    already_print_type = True
                source = inspect.getsource(func)
                ret += ' name: %s{\n' % name
                for line in source.split('\n'):
                    ret += '  %s\n' % line
                ret += ' }\n'
            if already_print_type:
                ret += '}\n'
        if not ret:
            ret = 'Not any customized hooks have been defined, you can set show_default=True to see the default hooks information'
        return ret

    def __getitem__(self, hook_type: str) -> OrderedDict:
        return self._registered_hooks[hook_type]

    def __repr__(self) -> str:
        return self.info(show_default=False)
