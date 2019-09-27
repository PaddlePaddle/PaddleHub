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
from paddlehub.commands.base_command import BaseCommand, ENTRY
from serving import app


class ServingCommand(BaseCommand):
    name = "serving"

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
        self.sub_parse.add_argument("--stop", action="store_true")
        self.parser.add_argument("--models", nargs="?")

    def execute(self, argv):
        # print(self.parser.parse_args())
        # print(self.parser.parse_args().models[0])
        # if self.parser.parse_args().models[0] == "lac":
        print("Serving starting...")
            # print("hehe")
        app.run()
        # else:
        #     print("Only supporting lac.")
        # print("123123")
        # print(argv)
        # print(self.parser)
        # args = self.parser.parse_args()
        # print(args)


command = ServingCommand.instance()
