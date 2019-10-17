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
import subprocess
import shlex
import os
import json
import paddlehub as hub
from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.serving import app_2


class ServingCommand(BaseCommand):
    name = "serving"
    module_list = []

    def __init__(self, name):
        super(ServingCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Start PaddleHub Serving."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s [COMMAND]' % (ENTRY, name),
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument("command")
        self.sub_parse = self.parser.add_mutually_exclusive_group(
            required=False)
        self.sub_parse.add_argument("--start", action="store_true")
        # self.sub_parse.add_argument("--stop", action="store_true")
        # self.sub_parse.add_argument("--show", action="store_true")
        self.parser.add_argument("--use_gpu", action="store_true")
        self.parser.add_argument("--modules", nargs="+")
        self.parser.add_argument("--config", "-c", nargs="+")

    @staticmethod
    def preinstall_modules(modules):
        configs = []
        if modules is not None:
            for module in modules:
                module_name = module if "==" not in module else \
                module.split("==")[0]
                module_version = None if "==" not in module else \
                module.split("==")[1]
                try:
                    m = hub.Module(name=module_name, version=module_version)
                    configs.append({
                        "module": module_name,
                        "version": m.version,
                        "category": str(m.type).split("/")[0].upper()
                    })
                except Exception as err:
                    pass
            return configs

    # @staticmethod
    # def preinstall_modules(modules):
    #     if modules is not None:
    #         for module in modules:
    #             module_name = module if "==" not in module else module.split("==")[0]
    #             module_version = None if "==" not in module else module.split("==")[1]
    #             try:
    #                 hub.Module(name=module_name, version=module_version)
    #
    #             except Exception as err:
    #                 pass

    @staticmethod
    def start_serving(module=None, use_gpu=False, config_file=None):
        if config_file is not None:
            config_file = config_file[0]
            if os.path.exists(config_file):
                with open(config_file, "r") as fp:
                    configs = json.load(fp)
                    module = [
                        str(i["module"]) + "==" + str(i["version"])
                        for i in configs
                    ]
                    module_info = ServingCommand.preinstall_modules(module)
                    for index in range(len(module_info)):
                        configs[index].update(module_info[index])
                    app_2.run(use_gpu, configs=configs)
            else:
                print("config_file ", config_file, "not exists.")
        elif module is not None:
            module_info = ServingCommand.preinstall_modules(module)
            [
                item.update({
                    "batch_size": 20,
                    "queue_size": 20
                }) for item in module_info
            ]
            app_2.run(use_gpu, configs=module_info)

    @staticmethod
    def stop_serving():
        print("Please kill this process by yourself.")
        return
        lsof_command = "lsof -i:8888"
        try:
            result = subprocess.check_output(shlex.split(lsof_command))
        except Exception as err:
            print("Serving has been stopped.")
            return
        result = result.splitlines()[1:]
        for item in result:
            process = item.split()
            ps_command = "ps " + process[1]
            res = subprocess.check_output(shlex.split(ps_command))
            if "gunicorn" in res:
                kill_command = "kill -9 " + process[1]
                subprocess.check_call(
                    shlex.split(kill_command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        print("Serving stop.")

    @staticmethod
    def show_modules():
        print("All models in use are as follows.")
        for module in ServingCommand.module_list:
            print(module)

    @staticmethod
    def show_help():
        str = "serving <option>\n"
        str += "\tManage PaddleHub-Serving.\n"
        str += "option:\n"
        str += "--start\n"
        str += "\tStart PaddleHub-Serving if specifies this parameter.\n"
        # str += "--stop\n"
        # str += "\tStop PaddleHub-Serving if specifies this parameter.\n"
        str += "--modules [module1==version, module2==version...]\n"
        str += "\tPre-install modules via this parameter list.\n"
        print(str)

    def execute(self, argv):
        args = self.parser.parse_args()
        print(args)
        if args.start is True:
            ServingCommand.start_serving(args.modules, args.use_gpu,
                                         args.config)
        # elif args.stop is True:
        #     ServingCommand.stop_serving()
        # elif args.show is True:
        #     ServingCommand.show_modules()
        else:
            ServingCommand.show_help()


command = ServingCommand.instance()
