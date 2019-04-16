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
import re

from paddlehub.common import utils
from paddlehub.common.downloader import default_downloader
from paddlehub.io.parser import yaml_parser
import paddlehub as hub

RESOURCE_LIST_FILE = "resource_list_file.yml"
CACHE_TIME = 60 * 10


class HubServer(object):
    def __init__(self, server_url=None):
        if not server_url:
            server_url = "https://paddlehub.bj.bcebos.com/"
        utils.check_url(server_url)
        self.server_url = server_url
        self._load_resource_list_file_if_valid()

    def resource_list_file_path(self):
        return os.path.join(hub.CACHE_HOME, RESOURCE_LIST_FILE)

    def _load_resource_list_file_if_valid(self):
        self.resource_list_file = {}
        if not os.path.exists(self.resource_list_file_path()):
            return False
        file_create_time = os.path.getctime(self.resource_list_file_path())
        now_time = time.time()

        # if file is out of date, remove it
        if now_time - file_create_time >= CACHE_TIME:
            os.remove(self.resource_list_file_path())
            return False
        for resource in yaml_parser.parse(
                self.resource_list_file_path())['resource_list']:
            for key in resource:
                if key not in self.resource_list_file:
                    self.resource_list_file[key] = []
                self.resource_list_file[key].append(resource[key])

        # if file format is invalid, remove it
        if "version" not in self.resource_list_file or "name" not in self.resource_list_file:
            self.resource_list_file = {}
            os.remove(self.resource_list_file_path())
            return False
        return True

    def search_resource(self, resource_key, resource_type=None, update=False):
        if update or not self.resource_list_file:
            self.request()

        if not self._load_resource_list_file_if_valid():
            return None

        match_resource_index_list = []
        for index, resource in enumerate(self.resource_list_file['name']):
            try:
                is_match = re.search(resource_key, resource)
                if is_match and (resource_type is None
                                 or self.resource_list_file['type'][index] ==
                                 resource_type):
                    match_resource_index_list.append(index)
            except:
                pass

        return [(self.resource_list_file['name'][index],
                 self.resource_list_file['type'][index],
                 self.resource_list_file['version'][index],
                 self.resource_list_file['summary'][index])
                for index in match_resource_index_list]

    def search_module(self, module_key, update=False):
        self.search_resource(
            resource_key=module_key, resource_type="Module", update=update)

    def search_model(self, module_key, update=False):
        self.search_resource(
            resource_key=module_key, resource_type="Model", update=update)

    def get_resource_url(self,
                         resource_name,
                         resource_type=None,
                         version=None,
                         update=False):
        if update or not self.resource_list_file:
            self.request()

        if not self._load_resource_list_file_if_valid():
            return {}

        resource_index_list = [
            index
            for index, resource in enumerate(self.resource_list_file['name'])
            if resource == resource_name and (
                resource_type is None
                or self.resource_list_file['type'][index] == resource_type)
        ]
        resource_version_list = [
            self.resource_list_file['version'][index]
            for index in resource_index_list
        ]
        #TODO(wuzewu): version sort method
        resource_version_list = sorted(resource_version_list)
        if not version:
            if not resource_version_list:
                return {}
            version = resource_version_list[-1]

        for index in resource_index_list:
            if self.resource_list_file['version'][index] == version:
                return {
                    'url': self.resource_list_file['url'][index],
                    'md5': self.resource_list_file['md5'][index],
                    'version': version
                }

        return {}

    def get_module_url(self, module_name, version=None, update=False):
        return self.get_resource_url(
            resource_name=module_name,
            resource_type="Module",
            version=version,
            update=update)

    def get_model_url(self, module_name, version=None, update=False):
        return self.get_resource_url(
            resource_name=module_name,
            resource_type="Model",
            version=version,
            update=update)

    def request(self):
        file_url = self.server_url + RESOURCE_LIST_FILE
        result, tips, self.resource_list_file = default_downloader.download_file(
            file_url, save_path=hub.CACHE_HOME)
        if not result:
            return False
        return True


default_hub_server = HubServer()
