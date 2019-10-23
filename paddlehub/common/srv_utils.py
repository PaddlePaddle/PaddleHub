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
import os
import requests
import time
import paddle
import socket
import json

from random import randint, seed

from paddlehub import version


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


def hub_request(api, params, extra=None):
    params['hub_version'] = version.hub_version
    params['paddle_version'] = paddle.__version__
    params["extra"] = json.dumps(extra)
    r = requests.get(api, params)
    return r.json()
