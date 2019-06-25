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

from paddlehub.common import utils
from paddlehub.common.hub_server import default_hub_server
from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.commands.cml_utils import TablePrinter


class SearchCommand(BaseCommand):
    name = "search"

    def __init__(self, name):
        super(SearchCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Search PaddleHub pretrained model through model keywords."
        self.parser = self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <key>' % (ENTRY, name),
            usage='%(prog)s',
            add_help=False)

    def execute(self, argv):
        if not argv:
            argv = ['.*']

        resource_name = argv[0]
        resource_list = default_hub_server.search_resource(
            resource_name, resource_type="Module")
        if utils.is_windows():
            placeholders = [20, 8, 8, 20]
        else:
            placeholders = [30, 8, 8, 25]
        tp = TablePrinter(
            titles=["ResourceName", "Type", "Version", "Summary"],
            placeholders=placeholders)
        for resource_name, resource_type, resource_version, resource_summary in resource_list:
            if resource_type == "Module":
                colors = ["yellow", None, None, None]
            else:
                colors = ["light_red", None, None, None]
            tp.add_line(
                contents=[
                    resource_name, resource_type, resource_version,
                    resource_summary
                ],
                colors=colors)
        print(tp.get_text())
        return True


command = SearchCommand.instance()
