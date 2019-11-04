# coding:utf-8
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
import os
import platform
import socket
import json
import paddlehub as hub
from paddlehub.commands.base_command import BaseCommand, ENTRY


class ServingCommand(BaseCommand):
    name = "serving"
    module_list = []

    def __init__(self, name):
        super(ServingCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Start a service for online predicting by using PaddleHub."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s [COMMAND]' % (ENTRY, name),
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument("command")
        self.parser.add_argument("sub_command")
        self.sub_parse = self.parser.add_mutually_exclusive_group(
            required=False)
        self.parser.add_argument(
            "--use_gpu", action="store_true", default=False)
        self.parser.add_argument(
            "--use_multiprocess", action="store_true", default=False)
        self.parser.add_argument("--modules", "-m", nargs="+")
        self.parser.add_argument("--config", "-c", nargs="+")
        self.parser.add_argument("--port", "-p", nargs="+", default=[8866])

    @staticmethod
    def is_port_occupied(ip, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, int(port)))
            s.shutdown(2)
            return True
        except:
            return False

    @staticmethod
    def preinstall_modules(modules):
        configs = []
        module_exist = {}
        if modules is not None:
            for module in modules:
                module_name = module if "==" not in module else \
                module.split("==")[0]
                module_version = None if "==" not in module else \
                module.split("==")[1]
                if module_exist.get(module_name, "") != "":
                    print(module_name, "==", module_exist.get(module_name),
                          " will be ignored cause new version is specified.")
                    configs.pop()
                module_exist.update({module_name: module_version})
                try:
                    m = hub.Module(name=module_name, version=module_version)
                    method_name = m.desc.attr.map.data['default_signature'].s
                    if method_name == "":
                        raise RuntimeError("{} cannot be use for "
                                           "predicting".format(module_name))
                    configs.append({
                        "module": module_name,
                        "version": m.version,
                        "category": str(m.type).split("/")[0].upper()
                    })
                except Exception as err:
                    print(err, ", start Hub-Serving unsuccessfully.")
                    exit(1)
            return configs

    @staticmethod
    def start_serving(args):
        config_file = args.config
        if config_file is not None:
            config_file = config_file[0]
            if os.path.exists(config_file):
                with open(config_file, "r") as fp:
                    configs = json.load(fp)
                    use_multiprocess = configs.get("use_multiprocess", False)
                    if use_multiprocess is True:
                        if platform.system() == "Windows":
                            print(
                                "Warning: Windows cannot use multiprocess working "
                                "mode, Hub-Serving will switch to single process mode"
                            )
                            from paddlehub.serving import app_single as app
                        else:
                            from paddlehub.serving import app
                    else:
                        from paddlehub.serving import app_single as app
                    use_gpu = configs.get("use_gpu", False)
                    port = configs.get("port", 8866)
                    if ServingCommand.is_port_occupied("127.0.0.1",
                                                       port) is True:
                        print("Port %s is occupied, please change it." % (port))
                        return False
                    configs = configs.get("modules_info")
                    module = [
                        str(i["module"]) + "==" + str(i["version"])
                        for i in configs
                    ]
                    module_info = ServingCommand.preinstall_modules(module)
                    for index in range(len(module_info)):
                        configs[index].update(module_info[index])
                    app.run(use_gpu, configs=configs, port=port)
            else:
                print("config_file ", config_file, "not exists.")
        else:
            if args.use_multiprocess is True:
                if platform.system() == "Windows":
                    print(
                        "Warning: Windows cannot use multiprocess working "
                        "mode, Hub-Serving will switch to single process mode")
                    from paddlehub.serving import app_single as app
                else:
                    from paddlehub.serving import app
            else:
                from paddlehub.serving import app_single as app
            module = args.modules
            if module is not None:
                use_gpu = args.use_gpu
                port = args.port[0]
                if ServingCommand.is_port_occupied("127.0.0.1", port) is True:
                    print("Port %s is occupied, please change it." % (port))
                    return False
                module_info = ServingCommand.preinstall_modules(module)
                [
                    item.update({
                        "batch_size": 1,
                        "queue_size": 20
                    }) for item in module_info
                ]
                app.run(use_gpu, configs=module_info, port=port)
            else:
                print("Lack of necessary parameters!")

    @staticmethod
    def show_help():
        str = "serving <option>\n"
        str += "\tManage PaddleHub-Serving.\n"
        str += "sub command:\n"
        str += "start\n"
        str += "\tStart PaddleHub-Serving if specifies this parameter.\n"
        str += "option:\n"
        str += "--modules/-m [module1==version, module2==version...]\n"
        str += "\tPre-install modules via this parameter list.\n"
        str += "--port/-p XXXX\n"
        str += "\tUse port XXXX for serving.\n"
        str += "--use_gpu\n"
        str += "\tUse gpu for predicting if specifies this parameter.\n"
        str += "--config/-c file_path\n"
        str += "\tUse configs in file to starting paddlehub serving."
        str += "Other parameter will be ignored if specifies this parameter.\n"
        print(str)

    def execute(self, argv):
        try:
            args = self.parser.parse_args()
        except:
            print("Please refer to the instructions below.")
            ServingCommand.show_help()
            return False
        if args.sub_command == "start":
            ServingCommand.start_serving(args)
        else:
            ServingCommand.show_help()


command = ServingCommand.instance()
