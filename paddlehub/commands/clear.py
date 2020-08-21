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
import shutil
from typing import List

import paddlehub.env as hubenv
from paddlehub.commands import register


def file_size_in_human_format(size: int) -> str:
    size = float(size)
    if size < 1024:
        return "%.1fB" % size
    elif size < 1024 * 1024:
        return "%.1fK" % (size / 1024)
    elif size < 1024 * 1024 * 1024:
        return "%.1fM" % (size / (1024 * 1024))
    else:
        return "%.1fG" % (size / (1024 * 1024 * 1024))


@register(name='hub.clear', description='Clear all cached data.')
class ClearCommand:
    def execute(self, argv: List) -> bool:
        total_file_size = 0.0
        total_file_cnt = 0

        for root, dirs, files in os.walk(hubenv.CACHE_HOME):
            total_file_cnt += len(files)
            total_file_cnt += len(dirs)
            for file in files:
                realpath = os.path.join(hubenv.CACHE_HOME, root, file)
                total_file_size += os.path.getsize(realpath)

            for dir in dirs:
                realdir = os.path.join(hubenv.CACHE_HOME, root, dir)
                total_file_size += os.path.getsize(realdir)

        shutil.rmtree(hubenv.CACHE_HOME)

        if total_file_cnt == 0:
            print('No cache to release.')
        else:
            print('Clear {} cached files.'.format(total_file_cnt))
            print('Free disk space {}.'.format(file_size_in_human_format(total_file_size)))

        return True
