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

from paddlehub import version


def get_stat_server():
    stat_srv = os.environ.get('HUB_SERVER_STAT_SRV')
    if not stat_srv:
        stat_srv = 'http://hub.paddlepaddle.org:8888/paddlehub/stat'
    return stat_srv


def hub_stat(argv):
    try:
        params = {'command': ' '.join(argv), 'version': version.hub_version}
        stat_api = get_stat_server()
        r = requests.get(stat_api, params=params, timeout=0.5)
    except:
        pass
