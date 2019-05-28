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
import os

from paddlehub.common import utils
from paddlehub.common.downloader import default_downloader
from paddlehub.common.hub_server import default_hub_server
from paddlehub.commands.base_command import BaseCommand, ENTRY


class DownloadCommand(BaseCommand):
    name = "download"

    def __init__(self, name):
        super(DownloadCommand, self).__init__(name)
        self.show_in_help = True
        self.description = "Download PaddlePaddle pretrained model/module files."
        self.parser = self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <model_name/module_name>' % (ENTRY, name),
            usage='%(prog)s [options]',
            add_help=False)
        # yapf: disable
        self.add_arg("--type",         str,  "All", "choice: Module/Model/All")
        self.add_arg('--output_path',  str,  ".",   "path to save the model/module" )
        self.add_arg('--uncompress',   bool, False,  "uncompress the download package or not" )
        # yapf: enable

    def execute(self, argv):
        if not argv:
            print("ERROR: Please provide the model/module name\n")
            self.help()
            return False
        mod_name = argv[0]
        mod_version = None if "==" not in mod_name else mod_name.split("==")[1]
        mod_name = mod_name if "==" not in mod_name else mod_name.split("==")[0]
        self.args = self.parser.parse_args(argv[1:])
        self.args.type = self.check_type(self.args.type)

        if self.args.type in ["Module", "Model"]:
            search_result = default_hub_server.get_resource_url(
                mod_name, resource_type=self.args.type, version=mod_version)
        else:
            search_result = default_hub_server.get_resource_url(
                mod_name, resource_type="Module", version=mod_version)
            self.args.type = "Module"
            if search_result == {}:
                search_result = default_hub_server.get_resource_url(
                    mod_name, resource_type="Model", version=mod_version)
                self.args.type = "Model"
        url = search_result.get('url', None)
        except_md5_value = search_result.get('md5', None)
        if not url:
            tips = "PaddleHub can't find model/module named %s" % mod_name
            if mod_version:
                tips += " with version %s" % mod_version
            tips += ". Please use the 'hub search' command to find the correct model/module name."
            print(tips)
            return True

        need_to_download_file = True
        file_name = os.path.basename(url)
        file = os.path.join(self.args.output_path, file_name)
        if os.path.exists(file):
            print("File %s already existed\nWait to check the MD5 value" %
                  file_name)
            file_md5_value = utils.md5_of_file(file)
            if except_md5_value == file_md5_value:
                print("MD5 check pass.")
                need_to_download_file = False
            else:
                print("MD5 check failed!\nDelete invalid file.")
                os.remove(file)

        if need_to_download_file:
            result, tips, file = default_downloader.download_file(
                url=url, save_path=self.args.output_path, print_progress=True)
            if not result:
                print(tips)
                return False

        if self.args.uncompress:
            result, tips, file = default_downloader.uncompress(
                file=file,
                dirname=self.args.output_path,
                delete_file=True,
                print_progress=True)
            print(tips)
            if self.args.type == "Model":
                os.rename(file, "./" + mod_name)
        return True

    def check_type(self, mod_type):
        mod_type = mod_type.lower()
        if mod_type == "module":
            mod_type = "Module"
        elif mod_type == "model":
            mod_type = "Model"
        else:
            mod_type = "All"
        return mod_type


command = DownloadCommand.instance()
