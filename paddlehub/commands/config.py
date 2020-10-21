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

import argparse
import json
import os
import re
import hashlib
import uuid
import time

from paddlehub.env import CONF_HOME
from paddlehub.commands import register
from paddlehub.utils.utils import md5

default_server_config = {
    "server_url": ["http://paddlepaddle.org.cn/paddlehub"],
    "resource_storage_server_url": "https://bj.bcebos.com/paddlehub-data/",
    "debug": False,
    "log_level": "DEBUG",
    "hub_name": md5(str(uuid.uuid1())[-12:]) + "-" + str(int(time.time()))
}


@register(name='hub.config', description='Configure PaddleHub.')
class ConfigCommand:
    @staticmethod
    def show_config():
        print("The current configuration is shown below.")
        with open(os.path.join(CONF_HOME, "config.json"), "r") as fp:
            print(json.dumps(json.load(fp), indent=4))

    @staticmethod
    def set_server_url(server_url):
        with open(os.path.join(CONF_HOME, "config.json"), "r") as fp:
            config = json.load(fp)
            re_str = r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\*\+,;=.]+$"
            if re.match(re_str, server_url) is not None:
                config["server_url"] = list([server_url])
                ConfigCommand.set_config(config)
            else:
                print("The format of the input url is invalid.")

    @staticmethod
    def set_config(config):
        with open(os.path.join(CONF_HOME, "config.json"), "w") as fp:
            fp.write(json.dumps(config))
        print("Set success! The current configuration is shown below.")
        print(json.dumps(config, indent=4))

    @staticmethod
    def set_log_level(level):
        level = str(level).upper()
        if level not in ["NOLOG", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            print("Allowed values include: " "NOLOG, DEBUG, INFO, WARNING, ERROR, CRITICAL")
            return
        with open(os.path.join(CONF_HOME, "config.json"), "r") as fp:
            current_config = json.load(fp)
        with open(os.path.join(CONF_HOME, "config.json"), "w") as fp:
            current_config["log_level"] = level
            fp.write(json.dumps(current_config))
            print("Set success! The current configuration is shown below.")
            print(json.dumps(current_config, indent=4))

    @staticmethod
    def show_help():
        str = "config <option>\n"
        str += "\tShow PaddleHub config without any option.\n"
        str += "option:\n"
        str += "reset\n"
        str += "\tReset config as default.\n"
        str += "server==[URL]\n"
        str += "\tSet PaddleHub Server url as [URL].\n"
        str += "log==[LEVEL]\n"
        str += "\tSet log level as [LEVEL:NOLOG, DEBUG, INFO, WARNING, ERROR, CRITICAL].\n"
        print(str)

    def execute(self, argv):
        if not argv:
            ConfigCommand.show_config()
        for arg in argv:
            if arg == "reset":
                ConfigCommand.set_config(default_server_config)
            elif arg.startswith("server=="):
                ConfigCommand.set_server_url(arg.split("==")[1])
            elif arg.startswith("log=="):
                ConfigCommand.set_log_level(arg.split("==")[1])
            else:
                ConfigCommand.show_help()
        return True
