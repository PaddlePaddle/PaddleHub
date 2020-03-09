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

from functools import cmp_to_key
import tarfile

import paddlehub as hub
from paddlehub.common import utils
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import MODULE_HOME
from paddlehub.common.cml_utils import TablePrinter
from paddlehub.common.logger import logger
from paddlehub.common import tmp_dir
from paddlehub.module import module_desc_pb2


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
                       module_name=None,
                       module_dir=None,
                       module_package=None,
                       module_version=None,
                       upgrade=False,
                       extra=None):
        md5_value = installed_module_version = None
        from_user_dir = True if module_dir else False
        with tmp_dir() as _dir:
            if module_name:
                self.all_modules(update=True)
                module_info = self.modules_dict.get(module_name, None)
                if module_info:
                    if not module_version or module_version == self.modules_dict[
                            module_name][1]:
                        module_dir = self.modules_dict[module_name][0]
                        module_tag = module_name if not module_version else '%s-%s' % (
                            module_name, module_version)
                        tips = "Module %s already installed in %s" % (
                            module_tag, module_dir)
                        return True, tips, self.modules_dict[module_name]

                search_result = hub.HubServer().get_module_url(
                    module_name, version=module_version, extra=extra)
                name = search_result.get('name', None)
                url = search_result.get('url', None)
                md5_value = search_result.get('md5', None)
                installed_module_version = search_result.get('version', None)
                if not url or (module_version is not None
                               and installed_module_version != module_version
                               ) or (name != module_name):
                    if hub.HubServer()._server_check() is False:
                        tips = "Request Hub-Server unsuccessfully, please check your network."
                        return False, tips, None
                    module_versions_info = hub.HubServer().search_module_info(
                        module_name)
                    if module_versions_info is not None and len(
                            module_versions_info) > 0:

                        if utils.is_windows():
                            placeholders = [20, 8, 14, 14]
                        else:
                            placeholders = [30, 8, 16, 16]
                        tp = TablePrinter(
                            titles=[
                                "ResourceName", "Version", "PaddlePaddle",
                                "PaddleHub"
                            ],
                            placeholders=placeholders)
                        module_versions_info.sort(
                            key=cmp_to_key(utils.sort_version_key))
                        for resource_name, resource_version, paddle_version, \
                            hub_version in module_versions_info:
                            colors = ["yellow", None, None, None]

                            tp.add_line(
                                contents=[
                                    resource_name, resource_version,
                                    utils.strflist_version(paddle_version),
                                    utils.strflist_version(hub_version)
                                ],
                                colors=colors)
                        tips = "The version of PaddlePaddle or PaddleHub " \
                               "can not match module, please upgrade your " \
                               "PaddlePaddle or PaddleHub according to the form " \
                               "below." + tp.get_text()
                    else:
                        tips = "Can't find module %s" % module_name
                        if module_version:
                            tips += " with version %s" % module_version
                    return False, tips, None

                result, tips, module_zip_file = default_downloader.download_file(
                    url=url,
                    save_path=_dir,
                    save_name=module_name,
                    replace=True,
                    print_progress=True)
                result, tips, module_dir = default_downloader.uncompress(
                    file=module_zip_file,
                    dirname=os.path.join(_dir, "tmp_module"),
                    delete_file=True,
                    print_progress=True)

            if module_package:
                with tarfile.open(module_package, "r:gz") as tar:
                    file_names = tar.getnames()
                    size = len(file_names) - 1
                    module_dir = os.path.join(_dir, file_names[0])
                    for index, file_name in enumerate(file_names):
                        tar.extract(file_name, _dir)
                    module_name = hub.Module(directory=module_dir).name

            if from_user_dir:
                module_name = hub.Module(directory=module_dir).name
                module_version = hub.Module(directory=module_dir).version
                self.all_modules(update=False)
                module_info = self.modules_dict.get(module_name, None)
                if module_info:
                    if module_version == module_info[1]:
                        module_dir = self.modules_dict[module_name][0]
                        module_tag = module_name if not module_version else '%s-%s' % (
                            module_name, module_version)
                        tips = "Module %s already installed in %s" % (
                            module_tag, module_dir)
                        return True, tips, self.modules_dict[module_name]

            if module_dir:
                if md5_value:
                    with open(
                            os.path.join(MODULE_HOME, module_dir, "md5.txt"),
                            "w") as fp:
                        fp.write(md5_value)

                save_path = os.path.join(MODULE_HOME, module_name)
                if save_path != module_dir:
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                    if from_user_dir:
                        shutil.copytree(module_dir, save_path)
                    else:
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
