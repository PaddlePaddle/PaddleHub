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

import os
from typing import List

from paddlehub.commands import register
from paddlehub.module.manager import LocalModuleManager
from paddlehub.utils import log, platform
from paddlehub.module.module import Module, InvalidHubModule


@register(name='hub.show', description='Show the information of PaddleHub module.')
class ShowCommand:
    def execute(self, argv: List) -> bool:
        if not argv:
            print("ERROR: You must give one module to show.")
            return False
        argv = argv[0]

        if os.path.exists(argv) and os.path.isdir(argv):
            try:
                module = Module.load(argv)
            except InvalidHubModule:
                print('{} is not a valid HubModule'.format(argv))
                return False
            except:
                print('Some exception occurred while loading the {}'.format(argv))
                return False
        else:
            module = LocalModuleManager().search(argv)
            if not module:
                print('{} is not existed!'.format(argv))
                return False

        widths = [15, 40] if platform.is_windows else [15, 50]
        aligns = ['^', '<']
        colors = ['cyan', '']
        table = log.Table(widths=widths, colors=colors, aligns=aligns)

        table.append('ModuleName', module.name)
        table.append('Version', str(module.version))
        table.append('Summary', module.summary)
        table.append('Author', module.author)
        table.append('Author-Email', module.author_email)
        table.append('Location', module.directory)
        print(table)
        return True
