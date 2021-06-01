#coding:utf-8
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
import os
from typing import List

from paddlehub.commands import register
from paddlehub.module.manager import LocalModuleManager
from paddlehub.server.server import module_server
from paddlehub.utils import log, platform
from paddlehub.server.server import CacheUpdater


@register(name='hub.search', description='Search PaddleHub pretrained model through model keywords.')
class SearchCommand:
    def execute(self, argv: List) -> bool:
        argv = '.*' if not argv else argv[0]

        widths = [20, 8, 30] if platform.is_windows() else [30, 8, 40]
        table = log.Table(widths=widths)
        table.append(*['ModuleName', 'Version', 'Summary'], aligns=['^', '^', '^'], colors=["blue", "blue", "blue"])
        CacheUpdater("hub_search", argv).start()
        results = module_server.search_module(name=argv)

        for result in results:
            if 'Module' == result['type']:
                table.append(result['name'], result['version'], result['summary'])

        print(table)
        return True
