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
from paddle_hub.tools import utils
import os


class LocalModuleManager:
    def __init__(self, base_path=None):
        self.base_path = base_path if base_path else os.path.expanduser('~')
        utils.check_path(self.base_path)
        self.local_hub_dir = os.path.join(self.base_path, ".hub")
        self.local_modules_dir = os.path.join(self.local_hub_dir, "modules")
        self.modules = []
        if not os.path.exists(self.local_modules_dir):
            utils.mkdir(self.local_modules_dir)
        elif os.path.isfile(self.local_modules_dir):
            #TODO(wuzewu): give wanring
            pass

    def check_module_valid(self, module_path):
        #TODO(wuzewu): code
        return True

    def all_modules(self, update=False):
        if not update and self.modules:
            return self.modules
        self.modules = []
        for sub_dir_name in os.listdir(self.local_modules_dir):
            sub_dir_path = os.path.join(self.local_modules_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path) and self.check_module_valid(
                    sub_dir_path):
                #TODO(wuzewu): get module name
                module_name = sub_dir_path
                self.modules.append(module_name)

        return self.modules

    def search_module(self, module_name, update=False):
        self.all_modules(update=update)
        return module_name in self.all_modules

    def install_module(self, upgrade=False):
        pass

    def uninstall_module(self):
        pass


default_manager = LocalModuleManager()
