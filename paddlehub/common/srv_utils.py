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
import requests
import paddle
import json
import time
import uuid
import os

from paddlehub import version
from paddlehub.common.dir import CONF_HOME
from paddlehub.common.decorator_utils import singleton
from paddlehub.common.utils import md5
from paddlehub.common.server_config import default_server_config


def uri_path(server_url, api):
    srv = server_url
    if server_url.endswith('/'):
        srv = server_url[:-1]
    if api.startswith('/'):
        srv += api
    else:
        api = '/' + api
        srv += api
    return srv


def hub_request(api, params, extra=None, timeout=8):
    params['hub_version'] = version.hub_version
    params['paddle_version'] = paddle.__version__
    params["extra"] = json.dumps(extra)
    r = requests.get(api, params, timeout=timeout)
    return r.json()


@singleton
class ConfigInfo(object):
    def __init__(self):
        self.filepath = os.path.join(CONF_HOME, "config.json")
        self.hub_name = None
        self.configs = None
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as fp:
                self.configs = json.load(fp)
                self.hub_name = self.configs.get("hub_name", None)

    def get_hub_name(self):
        if self.hub_name is None:
            self.hub_name = md5(str(uuid.uuid1())[-12:]) + "-" + str(
                int(time.time()))
            with open(self.filepath, "w") as fp:
                fp.write(json.dumps(default_server_config))

        return self.hub_name
