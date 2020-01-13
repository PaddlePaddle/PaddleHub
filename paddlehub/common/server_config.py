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
import time
import uuid
from paddlehub.common.utils import md5

HUB_SERVERS = ["http://paddlepaddle.org.cn/paddlehub"]
hub_name = md5(str(uuid.uuid1())[-12:]) + "-" + str(int(time.time()))

default_server_config = {
    "server_url": HUB_SERVERS,
    "resource_storage_server_url": "https://bj.bcebos.com/paddlehub-data/",
    "debug": False,
    "log_level": "DEBUG",
    "hub_name": hub_name
}
