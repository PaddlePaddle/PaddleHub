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

from paddlehub.common.arg_helper import add_argument, print_arguments

ENTRY = "hub"


class BaseCommand(object):
    command_dict = {}

    @classmethod
    def instance(cls):
        if cls.name in BaseCommand.command_dict:
            command = BaseCommand.command_dict[cls.name]
            if command.__class__.__name__ != cls.__name__:
                raise KeyError(
                    "Command dict already has a command %s with type %s" %
                    (cls.name, command.__class__))
            return command
        if not hasattr(cls, '_instance'):
            cls._instance = cls(cls.name)
        BaseCommand.command_dict[cls.name] = cls._instance
        return cls._instance

    def __init__(self, name):
        if hasattr(self.__class__, '_instance'):
            raise RuntimeError("Please use `instance()` to get Command object!")
        self.args = None
        self.name = name
        self.show_in_help = True
        self.description = ""

    def help(self):
        self.parser.print_help()

    def add_arg(self, argument, type="str", default=None, help=None):
        add_argument(
            argument=argument,
            type=type,
            default=default,
            help=help,
            argparser=self.parser)

    def print_args(self):
        print_arguments(self.args)

    def execute(self, argv):
        raise NotImplementedError("Base Command should not be execute!")
