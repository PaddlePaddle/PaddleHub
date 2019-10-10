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
import paddlehub as hub
from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.serving import app


class ServingCommand(BaseCommand):
    name = "serving"
    starting_flag = False
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

    @staticmethod
    def preinstall_modules(modules):
        if modules is not None:
            for module in modules:
                module_name = module if "==" not in module else module.split("==")[0]
                module_version = None if "==" not in module else module.split("==")[1]
                try:
                    hub.Module(name=module_name, version=module_version)
                    ServingCommand.module_list.append(module_name)
                except Exception as err:
                    pass

    @staticmethod
    def start_serving(module=None, use_gpu=False):
        if ServingCommand.starting_flag is True:
            print("Serving has been started.")
            return
        if module is not None:
            ServingCommand.preinstall_modules(module)
        try:
            ServingCommand.starting_flag = True
            app.run(use_gpu)
        except Exception as err:
            ServingCommand.starting_flag = False

    @staticmethod
    def stop_serving():
        print("Please kill this process by yourself.")
        return
        if ServingCommand.starting_flag is False:
            print("Serving has been stopped.")
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
                subprocess.check_call(shlex.split(kill_command), stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        print("Serving stop.")
        ServingCommand.starting_flag = False

    @staticmethod
    def show_modules():
        if ServingCommand.starting_flag is False:
            print("Serving has not been start.")
            return
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
        str += "--stop\n"
        str += "\tStop PaddleHub-Serving if specifies this parameter.\n"
        str += "--modules [module1==version, module2==version...]\n"
        str += "\tPre-install modules via this parameter list.\n"
        print(str)

    def execute(self, argv):
        args = self.parser.parse_args()
        if args.start is True:
            ServingCommand.start_serving(args.modules, args.use_gpu)
        # elif args.stop is True:
        #     ServingCommand.stop_serving()
        # elif args.show is True:
        #     ServingCommand.show_modules()
        else:
            ServingCommand.show_help()


command = ServingCommand.instance()
