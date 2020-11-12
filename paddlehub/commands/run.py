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

import argparse
import ast
import os
from typing import Any, List

from paddlehub.compat.module.module_v1 import ModuleV1
from paddlehub.commands import register
from paddlehub.module.manager import LocalModuleManager
from paddlehub.module.module import Module, InvalidHubModule
from paddlehub.server.server import CacheUpdater


@register(name='hub.run', description='Run the specific module.')
class RunCommand:
    def execute(self, argv: List) -> bool:
        if not argv:
            print('ERROR: You must give one module to run.')
            return False
        module_name = argv[0]
        CacheUpdater("hub_run", module_name).start()

        if os.path.exists(module_name) and os.path.isdir(module_name):
            try:
                module = Module.load(module_name)
            except InvalidHubModule:
                print('{} is not a valid HubModule'.format(module_name))
                return False
            except:
                print('Some exception occurred while loading the {}'.format(module_name))
                return False
        else:
            module = Module(name=module_name)

        if not module.is_runnable:
            print('ERROR! Module {} is not executable.'.format(module_name))
            return False

        if isinstance(module, ModuleV1):
            result = self.run_module_v1(module, argv[1:])
        else:
            result = module._run_func(argv[1:])

        print(result)
        return True

    def run_module_v1(self, module, argv: List) -> Any:
        parser = argparse.ArgumentParser(prog='hub run {}'.format(module.name), add_help=False)

        arg_input_group = parser.add_argument_group(title='Input options', description='Data feed into the module.')
        arg_config_group = parser.add_argument_group(
            title='Config options', description='Run configuration for controlling module behavior, optional.')

        arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help='whether use GPU for prediction')
        arg_config_group.add_argument('--batch_size', type=int, default=1, help='batch size for prediction')

        module_type = module.type.lower()
        if module_type.startswith('cv'):
            arg_input_group.add_argument(
                '--input_path', type=str, default=None, help='path of image/video to predict', required=True)
        else:
            arg_input_group.add_argument('--input_text', type=str, default=None, help='text to predict', required=True)

        args = parser.parse_args(argv)

        except_data_format = module.processor.data_format(module.default_signature)
        key = list(except_data_format.keys())[0]
        input_data = {key: [args.input_path] if module_type.startswith('cv') else [args.input_text]}

        return module(
            sign_name=module.default_signature, data=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)
