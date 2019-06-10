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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from paddlehub.commands.base_command import BaseCommand
from paddlehub.common.dir import CACHE_HOME


def file_num_in_dir(dirname):
    cnt = 1
    if os.path.isdir(dirname):
        for subfile in os.listdir(dirname):
            subfile = os.path.join(dirname, subfile)
            cnt += file_num_in_dir(subfile)
    return cnt


def file_size_in_human_format(size):
    size = float(size)
    if size < 1024:
        return "%.1fB" % size
    elif size < 1024 * 1024:
        return "%.1fK" % (size / 1024)
    elif size < 1024 * 1024 * 1024:
        return "%.1fM" % (size / (1024 * 1024))
    else:
        return "%.1fG" % (size / (1024 * 1024 * 1024))


class ClearCommand(BaseCommand):
    name = "clear"

    def __init__(self, name):
        super(ClearCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Clear all cached data."

    def cache_dir(self):
        return CACHE_HOME

    def execute(self, argv):
        result = True
        total_file_size = 0
        total_file_count = 0
        for rootdir, dirs, files in os.walk(self.cache_dir(), topdown=False):
            for filename in files:
                filename = os.path.join(rootdir, filename)
                try:
                    file_size = os.path.getsize(filename)
                    file_count = file_num_in_dir(filename)
                    os.remove(filename)
                    total_file_size += file_size
                    total_file_count += file_count
                except Exception as e:
                    result = False
            for dirname in dirs:
                dirname = os.path.join(rootdir, dirname)
                try:
                    dir_size = os.path.getsize(dirname)
                    file_count = file_num_in_dir(dirname)
                    os.rmdir(dirname)
                    total_file_size += dir_size
                    total_file_count += file_count
                except Exception as e:
                    result = False
        if total_file_count != 0:
            print("Clear %d cached files." % total_file_count)
            print("Free disk space %s." %
                  file_size_in_human_format(total_file_size))
        else:
            if result:
                print("No cache to release.")
            else:
                print("Clear cache failed!")

        return result


command = ClearCommand.instance()
