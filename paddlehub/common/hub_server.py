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
import time

from paddlehub.common import utils
from paddlehub.common.downloader import default_downloader
from paddlehub.io.reader import csv_reader
import paddlehub as hub

MODULE_LIST_FILE = "module_list_file.csv"
MODEL_LIST_FILE = "model_list_file.csv"
CACHE_TIME = 60 * 10


class HubServer:
    def __init__(self, server_url=None):
        if not server_url:
            server_url = "https://paddlehub.bj.bcebos.com/"
        utils.check_url(server_url)
        self.server_url = server_url
        self._load_module_list_file_if_valid()
        self._load_model_list_file_if_valid()

    def module_list_file_path(self):
        return os.path.join(hub.CACHE_HOME, MODULE_LIST_FILE)

    def model_list_file_path(self):
        return os.path.join(hub.CACHE_HOME, MODEL_LIST_FILE)

    def _load_model_list_file_if_valid(self):
        self.model_list_file = {}
        if not os.path.exists(self.model_list_file_path()):
            return False
        file_create_time = os.path.getctime(self.model_list_file_path())
        now_time = time.time()

        # if file is out of date, remove it
        if now_time - file_create_time >= CACHE_TIME:
            os.remove(self.model_list_file_path())
            return False
        self.model_list_file = csv_reader.read(self.model_list_file_path())

        # if file do not contain necessary data, remove it
        if "version" not in self.model_list_file or "model_name" not in self.model_list_file:
            self.model_list_file = {}
            os.remove(self.model_list_file_path())
            return False
        return True

    def _load_module_list_file_if_valid(self):
        self.module_list_file = {}
        if not os.path.exists(self.module_list_file_path()):
            return False
        file_create_time = os.path.getctime(self.module_list_file_path())
        now_time = time.time()

        # if file is out of date, remove it
        if now_time - file_create_time >= CACHE_TIME:
            os.remove(self.module_list_file_path())
            return False
        self.module_list_file = csv_reader.read(self.module_list_file_path())

        # if file do not contain necessary data, remove it
        if "version" not in self.module_list_file or "module_name" not in self.module_list_file:
            self.module_list_file = {}
            os.remove(self.module_list_file_path())
            return False
        return True

    def search_module(self, module_key, update=False):
        if update or not self.module_list_file:
            self.request()

        match_module_index_list = [
            index
            for index, module in enumerate(self.module_list_file['module_name'])
            if module_key in module
        ]

        return [(self.module_list_file['module_name'][index],
                 self.module_list_file['version'][index])
                for index in match_module_index_list]

    def search_model(self, model_key, update=False):
        if update or not self.model_list_file:
            self.request_model()

        match_model_index_list = [
            index
            for index, model in enumerate(self.model_list_file['model_name'])
            if model_key in model
        ]

        return [(self.model_list_file['model_name'][index],
                 self.model_list_file['version'][index])
                for index in match_model_index_list]

    def get_module_url(self, module_name, version=None, update=False):
        if update or not self.module_list_file:
            self.request()

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

    def get_model_url(self, model_name, version=None, update=False):
        if update or not self.model_list_file:
            self.request_model()

        model_index_list = [
            index
            for index, model in enumerate(self.model_list_file['model_name'])
            if model == model_name
        ]
        model_version_list = [
            self.model_list_file['version'][index] for index in model_index_list
        ]
        #TODO(wuzewu): version sort method
        model_version_list = sorted(model_version_list)
        if not version:
            if not model_version_list:
                return None
            version = model_version_list[-1]

        for index in model_index_list:
            if self.model_list_file['version'][index] == version:
                return self.model_list_file['url'][index]

        return None

    def request(self):
        file_url = self.server_url + MODULE_LIST_FILE
        result, tips, self.module_list_file = default_downloader.download_file(
            file_url, save_path=hub.CACHE_HOME)
        if not result:
            return False
        return self._load_module_list_file_if_valid()

    def request_model(self):
        file_url = self.server_url + MODEL_LIST_FILE
        result, tips, self.model_list_file = default_downloader.download_file(
            file_url, save_path=hub.CACHE_HOME)
        if not result:
            return False
        return self._load_model_list_file_if_valid()


default_hub_server = HubServer()
