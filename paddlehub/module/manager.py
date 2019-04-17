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

import os
import shutil

from paddlehub.common import utils
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import MODULE_HOME
import paddlehub as hub


class LocalModuleManager(object):
    def __init__(self, module_home=None):
        self.local_modules_dir = module_home if module_home else MODULE_HOME
        self.modules_dict = {}
        if not os.path.exists(self.local_modules_dir):
            utils.mkdir(self.local_modules_dir)
        elif os.path.isfile(self.local_modules_dir):
            #TODO(wuzewu): give wanring
            pass

    def check_module_valid(self, module_path):
        #TODO(wuzewu): code
        return True

    def all_modules(self, update=False):
        if not update and self.modules_dict:
            return self.modules_dict
        self.modules_dict = {}
        for sub_dir_name in os.listdir(self.local_modules_dir):
            sub_dir_path = os.path.join(self.local_modules_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path) and self.check_module_valid(
                    sub_dir_path):
                #TODO(wuzewu): get module name
                module_name = sub_dir_name
                self.modules_dict[module_name] = sub_dir_path

        return self.modules_dict

    def search_module(self, module_name, update=False):
        self.all_modules(update=update)
        return self.modules_dict.get(module_name, None)

    def install_module(self, module_name, module_version=None, upgrade=False):
        self.all_modules(update=True)
        if module_name in self.modules_dict:
            module_dir = self.modules_dict[module_name]
            tips = "Module %s already installed in %s" % (module_name,
                                                          module_dir)
            return True, tips, module_dir
        search_result = hub.default_hub_server.get_module_url(
            module_name, version=module_version)
        url = search_result.get('url', None)
        md5_value = search_result.get('md5', None)
        installed_module_version = search_result.get('version', None)
        #TODO(wuzewu): add compatibility check
        if not url:
            tips = "Can't find module %s" % module_name
            if module_version:
                tips += " with version %s" % module_version
            return False, tips, None

        result, tips, module_zip_file = default_downloader.download_file(
            url=url,
            save_path=hub.CACHE_HOME,
            save_name=module_name,
            replace=True,
            print_progress=True)
        result, tips, module_dir = default_downloader.uncompress(
            file=module_zip_file,
            dirname=MODULE_HOME,
            delete_file=True,
            print_progress=True)

        save_path = os.path.join(MODULE_HOME, module_name)
        shutil.move(module_dir, save_path)
        module_dir = save_path

        if module_dir:
            tips = "Successfully installed %s" % module_name
            if installed_module_version:
                tips += "-%s" % installed_module_version
            return True, tips, module_dir
        tips = "Download %s-%s failed" % (module_name, module_version)
        return False, tips, module_dir

    def uninstall_module(self, module_name):
        self.all_modules(update=True)
        if not module_name in self.modules_dict:
            tips = "%s is not installed" % module_name
            return True, tips
        tips = "Successfully uninstalled %s" % module_name
        module_dir = self.modules_dict[module_name]
        shutil.rmtree(module_dir)
        return True, tips


default_module_manager = LocalModuleManager()
