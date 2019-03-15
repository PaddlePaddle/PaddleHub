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

import os
import sys
import hashlib
import requests
import tempfile
import tarfile
from paddle_hub.tools import utils
from paddle_hub.tools.logger import logger
from paddle_hub.data.reader import csv_reader

__all__ = ['MODULE_HOME', 'downloader', 'md5file', 'Downloader']

# TODO(ZeyuChen) add environment varialble to set MODULE_HOME
MODULE_HOME = os.path.expanduser('~')
MODULE_HOME = os.path.join(MODULE_HOME, ".hub")
MODULE_HOME = os.path.join(MODULE_HOME, "modules")


# When running unit tests, there could be multiple processes that
# trying to create MODULE_HOME directory simultaneously, so we cannot
# use a if condition to check for the existence of the directory;
# instead, we use the filesystem as the synchronization mechanism by
# catching returned errors.
def must_mkdirs(path):
    try:
        os.makedirs(MODULE_HOME)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


class Downloader:
    def __init__(self, module_home=None):
        self.module_home = module_home if module_home else MODULE_HOME
        self.module_list_file = []

    def download_file(self, url, save_path=None, save_name=None, retry_limit=3):
        module_name = url.split("/")[-2]
        save_path = self.module_home if save_path is None else save_path
        if not os.path.exists(save_path):
            utils.mkdir(save_path)
        save_name = url.split('/')[-1] if save_name is None else save_name
        file_name = os.path.join(save_path, save_name)
        retry_times = 0
        while not (os.path.exists(file_name)):
            if os.path.exists(file_name):
                logger.info("file md5", md5file(file_name))
            if retry_times < retry_limit:
                retry_times += 1
            else:
                raise RuntimeError(
                    "Cannot download {0} within retry limit {1}".format(
                        url, retry_limit))
            logger.info(
                "Cache file %s not found, downloading %s" % (file_name, url))
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
                    for data in r.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write(
                            "\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                        sys.stdout.flush()

        logger.info("file %s download completed!" % (file_name))
        return file_name

    def uncompress(self, file, dirname=None, delete_file=False):
        dirname = os.path.dirname(file) if dirname is None else dirname
        with tarfile.open(file, "r:gz") as tar:
            file_names = tar.getnames()
            logger.info(file_names)
            module_dir = os.path.join(dirname, file_names[0])
            for file_name in file_names:
                tar.extract(file_name, dirname)

        if delete_file:
            os.remove(file)

        return module_dir

    def download_file_and_uncompress(self,
                                     url,
                                     save_path=None,
                                     save_name=None,
                                     retry_limit=3,
                                     delete_file=True):
        file = self.download_file(
            url=url,
            save_path=save_path,
            save_name=save_name,
            retry_limit=retry_limit)
        return self.uncompress(file, delete_file=delete_file)

    def search_module(self, module_name):
        if not self.module_list_file:
            #TODO(wuzewu): download file in tmp directory
            self.module_list_file = self.download_file(
                url="https://paddlehub.bj.bcebos.com/module_file_list.csv")
            self.module_list_file = csv_reader.read(self.module_list_file)

        match_module_index_list = [
            index
            for index, module in enumerate(self.module_list_file['module_name'])
            if module_name in module
        ]

        return [(self.module_list_file['module_name'][index],
                 self.module_list_file['version'][index])
                for index in match_module_index_list]

    def get_module_url(self, module_name, version=None):
        if not self.module_list_file:
            #TODO(wuzewu): download file in tmp directory
            self.module_list_file = self.download_file(
                url="https://paddlehub.bj.bcebos.com/module_file_list.csv")
            self.module_list_file = csv_reader.read(self.module_list_file)

        module_index_list = [
            index
            for index, module in enumerate(self.module_list_file['module_name'])
            if module == module_name
        ]
        module_version_list = [
            self.module_list_file['version'][index]
            for index in module_index_list
        ]
        #TODO(wuzewu): version sort method
        module_version_list = sorted(module_version_list)
        if not version:
            if not module_version_list:
                return None
            version = module_version_list[-1]

        for index in module_index_list:
            if self.module_list_file['version'][index] == version:
                return self.module_list_file['url'][index]

        return None


default_downloader = Downloader()
