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

import os
import shutil

from paddlehub.common import utils
from paddlehub.common import srv_utils
from paddlehub.common.downloader import default_downloader
from paddlehub.common.hub_server import default_hub_server
from paddlehub.common.dir import MODULE_HOME
from paddlehub.module import module_desc_pb2
import paddlehub as hub
from paddlehub.common.logger import logger


class LocalModuleManager(object):
    def __init__(self, module_home=None):
        self.local_modules_dir = module_home if module_home else MODULE_HOME
        self.modules_dict = {}
        if not os.path.exists(self.local_modules_dir):
            utils.mkdir(self.local_modules_dir)
        elif os.path.isfile(self.local_modules_dir):
            raise ValueError("Module home should be a folder, not a file")

    def check_module_valid(self, module_path):
        try:
            desc_pb_path = os.path.join(module_path, 'module_desc.pb')
            if os.path.exists(desc_pb_path) and os.path.isfile(desc_pb_path):
                info = {}
                desc = module_desc_pb2.ModuleDesc()
                with open(desc_pb_path, "rb") as fp:
                    desc.ParseFromString(fp.read())
                info['version'] = desc.attr.map.data["module_info"].map.data[
                    "version"].s
                return True, info
            else:
                logger.warning(
                    "%s does not exist, the module will be reinstalled" %
                    desc_pb_path)
        except:
            pass
        return False, None

    def all_modules(self, update=False):
        if not update and self.modules_dict:
            return self.modules_dict
        self.modules_dict = {}
        for sub_dir_name in os.listdir(self.local_modules_dir):
            sub_dir_path = os.path.join(self.local_modules_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path):
                valid, info = self.check_module_valid(sub_dir_path)
                if valid:
                    module_name = sub_dir_name
                    self.modules_dict[module_name] = (sub_dir_path,
                                                      info['version'])
        return self.modules_dict

    def search_module(self, module_name, module_version=None, update=False):
        self.all_modules(update=update)
        return self.modules_dict.get(module_name, None)

    def install_module(self,
                       module_name,
                       module_version=None,
                       upgrade=False,
                       extra=None):
        self.all_modules(update=True)
        module_info = self.modules_dict.get(module_name, None)
        if module_info:
            if not module_version or module_version == self.modules_dict[
                    module_name][1]:
                module_dir = self.modules_dict[module_name][0]
                module_tag = module_name if not module_version else '%s-%s' % (
                    module_name, module_version)
                tips = "Module %s already installed in %s" % (module_tag,
                                                              module_dir)
                return True, tips, self.modules_dict[module_name]

        search_result = hub.default_hub_server.get_module_url(
            module_name, version=module_version, extra=extra)
        name = search_result.get('name', None)
        url = search_result.get('url', None)
        md5_value = search_result.get('md5', None)
        installed_module_version = search_result.get('version', None)
        if not url or (module_version is not None and installed_module_version
                       != module_version) or (name != module_name):
            if default_hub_server._server_check() is False:
                tips = "Request Hub-Server unsuccessfully, please check your network."
            else:
                tips = "Can't find module %s" % module_name
                if module_version:
                    tips += " with version %s" % module_version
                module_tag = module_name if not module_version else '%s-%s' % (
                    module_name, module_version)
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

        if module_dir:
            with open(os.path.join(MODULE_HOME, module_dir, "md5.txt"),
                      "w") as fp:
                fp.write(md5_value)
            save_path = os.path.join(MODULE_HOME, module_name)
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            shutil.move(module_dir, save_path)
            module_dir = save_path
            tips = "Successfully installed %s" % module_name
            if installed_module_version:
                tips += "-%s" % installed_module_version
            return True, tips, (module_dir, installed_module_version)
        tips = "Download %s-%s failed" % (module_name, module_version)
        return False, tips, module_dir

    def uninstall_module(self, module_name, module_version=None):
        self.all_modules(update=True)
        if not module_name in self.modules_dict:
            tips = "%s is not installed" % module_name
            return True, tips
        if module_version and module_version != self.modules_dict[module_name][
                1]:
            tips = "%s-%s is not installed" % (module_name, module_version)
            return True, tips
        tips = "Successfully uninstalled %s" % module_name
        if module_version:
            tips += '-%s' % module_version
        module_dir = self.modules_dict[module_name][0]
        shutil.rmtree(module_dir)
        return True, tips


default_module_manager = LocalModuleManager()
