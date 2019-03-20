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
from paddle_hub.tools import utils
from paddle_hub.tools.downloader import default_downloader
from paddle_hub.data.reader import csv_reader
import os
import paddle_hub as hub

MODULE_LIST_FILE = "module_file_list.csv"


class HubServer:
    def __init__(self, server_url=None):
        if not server_url:
            server_url = "https://paddlehub.bj.bcebos.com/"
        utils.check_url(server_url)
        self.server_url = server_url
        self.module_file_list = []

    def search_module(self, module_key, update=False):
        if update or not self.module_file_list:
            self.request()

        match_module_index_list = [
            index
            for index, module in enumerate(self.module_list_file['module_name'])
            if module_key in module
        ]

        return [(self.module_list_file['module_name'][index],
                 self.module_list_file['version'][index])
                for index in match_module_index_list]

    def get_module_url(self, module_name, version=None, update=False):
        if update or not self.module_file_list:
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

    def request(self):
        file_url = self.server_url + MODULE_LIST_FILE
        self.module_list_file = default_downloader.download_file(
            file_url, save_path=hub.CACHE_HOME)
        self.module_list_file = csv_reader.read(self.module_list_file)
        return True


default_hub_server = HubServer()
