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
import ast

import paddlehub.config as hubconf
from paddlehub.env import CONF_HOME
from paddlehub.commands import register


@register(name='hub.config', description='Configure PaddleHub.')
class ConfigCommand:
    @staticmethod
    def show_config():
        print("The current configuration is shown below.")
        print(hubconf)

    @staticmethod
    def show_help():
        str = "config <option>\n"
        str += "\tShow PaddleHub config without any option.\n"
        str += "option:\n"
        str += "reset\n"
        str += "\tReset config as default.\n\n"
        str += "server==[URL]\n"
        str += "\tSet PaddleHub Server url as [URL].\n\n"
        str += "log.level==[LEVEL]\n"
        str += "\tSet log level.\n\n"
        str += "log.enable==True|False\n"
        str += "\tEnable or disable logger in PaddleHub.\n"
        print(str)

    def execute(self, argv):
        if not argv:
            ConfigCommand.show_config()
        for arg in argv:
            if arg == "reset":
                hubconf.reset()
                print(hubconf)
            elif arg.startswith("server=="):
                hubconf.server = arg.split("==")[1]
            elif arg.startswith("log.level=="):
                hubconf.log_level = arg.split("==")[1]
            elif arg.startswith("log.enable=="):
                hubconf.log_enable = ast.literal_eval(arg.split("==")[1])
            else:
                ConfigCommand.show_help()
        return True
