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
import time
import re
import requests
import json
import yaml
import random
import threading

from random import randint
from paddlehub.common import utils, srv_utils
from paddlehub.common.downloader import default_downloader
from paddlehub.common.server_config import default_server_config
from paddlehub.io.parser import yaml_parser
from paddlehub.common.lock import lock
from paddlehub.common.dir import CONF_HOME, CACHE_HOME

RESOURCE_LIST_FILE = "resource_list_file.yml"
CACHE_TIME = 60 * 10


class HubServer(object):
    def __init__(self, config_file_path=None):
        if not config_file_path:
            config_file_path = os.path.join(CONF_HOME, 'config.json')
        if not os.path.exists(CONF_HOME):
            utils.mkdir(CONF_HOME)
        if not os.path.exists(config_file_path):
            with open(config_file_path, 'w+') as fp:
                lock.flock(fp, lock.LOCK_EX)
                fp.write(json.dumps(default_server_config))
                lock.flock(fp, lock.LOCK_UN)

        with open(config_file_path) as fp:
            self.config = json.load(fp)
        fp_lock = open(config_file_path)
        lock.flock(fp_lock, lock.LOCK_EX)

        utils.check_url(self.config['server_url'])
        self.server_url = self.config['server_url']
        self.request()
        self._load_resource_list_file_if_valid()
        lock.flock(fp_lock, lock.LOCK_UN)

    def get_server_url(self):
        random.seed(int(time.time()))
        HS_ENV = os.environ.get('HUB_SERVER')
        if HS_ENV:
            HUB_SERVERS = HS_ENV.split(';')
            return HUB_SERVERS[random.randint(0, len(HUB_SERVERS) - 1)]
        return self.server_url[random.randint(0, len(self.server_url) - 1)]

    def resource_list_file_path(self):
        return os.path.join(CACHE_HOME, RESOURCE_LIST_FILE)

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

    def search_resource(self,
                        resource_key,
                        resource_type=None,
                        update=False,
                        extra=None):
        try:
            payload = {'word': resource_key}
            if resource_type:
                payload['type'] = resource_type
            api_url = srv_utils.uri_path(self.get_server_url(), 'search')
            r = srv_utils.hub_request(api_url, payload, extra=extra)
            if r['status'] == 0 and len(r['data']) > 0:
                return [(item['name'], item['type'], item['version'],
                         item['summary']) for item in r['data']]
        except:
            if self.config.get('debug', False):
                raise
            else:
                pass

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
                         update=False,
                         extra=None):
        try:
            payload = {'word': resource_name}
            if resource_type:
                payload['type'] = resource_type
            if version:
                payload['version'] = version
            api_url = srv_utils.uri_path(self.get_server_url(), 'search')
            r = srv_utils.hub_request(api_url, payload, extra)
            if r['status'] == 0 and len(r['data']) > 0:
                for item in r['data']:
                    if resource_name.lower() == item['name'].lower():
                        return item
                return r['data'][0]
        except:
            if self.config.get('debug', False):
                raise
            else:
                pass

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

    def get_module_url(self,
                       module_name,
                       version=None,
                       update=False,
                       extra=None):
        return self.get_resource_url(
            resource_name=module_name,
            resource_type="Module",
            version=version,
            update=update,
            extra=extra)

    def get_model_url(self, module_name, version=None, update=False,
                      extra=None):
        return self.get_resource_url(
            resource_name=module_name,
            resource_type="Model",
            version=version,
            update=update,
            extra=extra)

    def request(self):
        if not os.path.exists(CACHE_HOME):
            utils.mkdir(CACHE_HOME)
        try:
            r = requests.get(self.get_server_url() + '/' + 'search')
            data = json.loads(r.text)
            cache_path = os.path.join(CACHE_HOME, RESOURCE_LIST_FILE)
            with open(cache_path, 'w+') as fp:
                yaml.safe_dump({'resource_list': data['data']}, fp)
            return True
        except:
            if self.config.get('debug', False):
                raise
            else:
                pass
        try:
            file_url = self.config[
                'resource_storage_server_url'] + RESOURCE_LIST_FILE
            result, tips, self.resource_list_file = default_downloader.download_file(
                file_url, save_path=CACHE_HOME, replace=True)
            if not result:
                return False
        except:
            return False
        return True

    def _server_check(self):
        try:
            r = requests.get(self.get_server_url() + '/search')
            if r.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    def server_check(self):
        if self._server_check() is True:
            print("Request Hub-Server successfully.")
        else:
            print("Request Hub-Server unsuccessfully.")


class CacheUpdater(threading.Thread):
    def __init__(self, module, version=None):
        threading.Thread.__init__(self)
        self.module = module
        self.version = version

    def update_resource_list_file(self, module, version=None):
        payload = {'word': module}
        if version:
            payload['version'] = version
        api_url = srv_utils.uri_path(default_hub_server.get_server_url(),
                                     'search')
        cache_path = os.path.join(CACHE_HOME, RESOURCE_LIST_FILE)
        extra = {
            "command": "update_cache",
            "mtime": os.stat(cache_path).st_mtime
        }
        try:
            r = srv_utils.hub_request(api_url, payload, extra)
        except Exception as err:
            pass
        if r.get("update_cache", 0) == 1:
            with open(cache_path, 'w+') as fp:
                yaml.safe_dump({'resource_list': r['data']}, fp)

    def run(self):
        self.update_resource_list_file(self.module, self.version)


def server_check():
    default_hub_server.server_check()


default_hub_server = HubServer()
