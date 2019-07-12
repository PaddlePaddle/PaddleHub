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
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import print_function

import shutil
import os
import sys
import time
import hashlib
import requests
import tarfile

from paddlehub.common import utils
from paddlehub.common.logger import logger

__all__ = ['Downloader']
FLUSH_INTERVAL = 0.1

lasttime = time.time()


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


class Downloader(object):
    def download_file(self,
                      url,
                      save_path,
                      save_name=None,
                      retry_limit=3,
                      print_progress=False,
                      replace=False):
        if not os.path.exists(save_path):
            utils.mkdir(save_path)
        save_name = url.split('/')[-1] if save_name is None else save_name
        file_name = os.path.join(save_path, save_name)
        retry_times = 0

        if replace and os.path.exists(file_name):
            os.remove(file_name)

        while not (os.path.exists(file_name)):
            if retry_times < retry_limit:
                retry_times += 1
            else:
                tips = "Cannot download {0} within retry limit {1}".format(
                    url, retry_limit)
                return False, tips, None
            r = requests.get(url, stream=True)
            total_length = r.headers.get('content-length')

            if total_length is None:
                with open(file_name, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            else:
                #TODO(ZeyuChen) upgrade to tqdm process
                with open(file_name, 'wb') as f:
                    dl = 0
                    total_length = int(total_length)
                    starttime = time.time()
                    if print_progress:
                        print("Downloading %s" % save_name)
                    for data in r.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        if print_progress:
                            done = int(50 * dl / total_length)
                            progress(
                                "[%-50s] %.2f%%" %
                                ('=' * done, float(dl / total_length * 100)))
                if print_progress:
                    progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

        tips = "File %s download completed!" % (file_name)
        return True, tips, file_name

    def uncompress(self,
                   file,
                   dirname=None,
                   delete_file=False,
                   print_progress=False):
        dirname = os.path.dirname(file) if dirname is None else dirname
        if print_progress:
            print("Uncompress %s" % file)
        with tarfile.open(file, "r:gz") as tar:
            file_names = tar.getnames()
            size = len(file_names) - 1
            module_dir = os.path.join(dirname, file_names[0])
            for index, file_name in enumerate(file_names):
                if print_progress:
                    done = int(50 * float(index) / size)
                    progress("[%-50s] %.2f%%" % ('=' * done,
                                                 float(index / size * 100)))
                tar.extract(file_name, dirname)

            if print_progress:
                progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)
        if delete_file:
            os.remove(file)

        return True, "File %s uncompress completed!" % file, module_dir

    def download_file_and_uncompress(self,
                                     url,
                                     save_path,
                                     save_name=None,
                                     retry_limit=3,
                                     delete_file=True,
                                     print_progress=False,
                                     replace=False):
        result, tips_1, file = self.download_file(
            url=url,
            save_path=save_path,
            save_name=save_name,
            retry_limit=retry_limit,
            print_progress=print_progress,
            replace=replace)
        if not result:
            return result, tips_1, file
        result, tips_2, file = self.uncompress(
            file, delete_file=delete_file, print_progress=print_progress)
        if not result:
            return result, tips_2, file
        if save_name:
            save_name = os.path.join(save_path, save_name)
            shutil.move(file, save_name)
            return result, "%s\n%s" % (tips_1, tips_2), save_name
        return result, "%s\n%s" % (tips_1, tips_2), file


default_downloader = Downloader()
