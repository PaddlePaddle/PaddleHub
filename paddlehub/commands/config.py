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

import argparse
import json
import os
import re

from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.common.dir import CONF_HOME
from paddlehub.common.server_config import default_server_config


class ConfigCommand(BaseCommand):
    name = "config"

    def __init__(self, name):
        super(ConfigCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Configure PaddleHub."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s [COMMAND]' % (ENTRY, name),
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument("command")
        self.parser.add_argument("option", nargs="?")
        self.parser.add_argument("value", nargs="?")

    @staticmethod
    def show_config():
        print("The current configuration is shown below.")
        with open(os.path.join(CONF_HOME, "config.json"), "r") as fp:
            print(json.dumps(json.load(fp), indent=4))

    @staticmethod
    def set_server_url(server_url):
        with open(os.path.join(CONF_HOME, "config.json"), "r") as fp:
            config = json.load(fp)
            re_str = "^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\*\+,;=.]+$"
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
        if level not in [
                "NOLOG", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ]:
            print("Allowed values include: "
                  "NOLOG, DEBUG, INFO, WARNING, ERROR, CRITICAL")
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
        str += "\tShow hub server config without any option.\n"
        str += "option:\n"
        str += "reset\n"
        str += "\tReset config as default.\n"
        str += "server==[URL]\n"
        str += "\tSet hub server url as [URL].\n"
        str += "log==[LEVEL]\n"
        str += "\tSet log level as [LEVEL:NOLOG, DEBUG, INFO, WARNING, ERROR, CRITICAL].\n"
        print(str)

    def execute(self, argv):
        args = self.parser.parse_args()
        if args.option is None:
            ConfigCommand.show_config()
        elif args.option == "reset":
            ConfigCommand.set_config(default_server_config)
        elif args.option.startswith("server=="):
            ConfigCommand.set_server_url(args.option.split("==")[1])
        elif args.option.startswith("log=="):
            ConfigCommand.set_log_level(args.option.split("==")[1])
        else:
            ConfigCommand.show_help()
        return True


command = ConfigCommand.instance()
