# coding: utf-8
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
import paddlehub as hub


class ModuleManager(object):
    def __init__(self):
        self.modules = {}

    def load_module(self, modules=[]):
        for name in modules:
            self.modules.update({name: hub.Module(name)})
            print("Loading %s successful." % name)

    def get_module(self, name):
        if name in self.modules.keys():
            return self.modules[name]
        else:
            return hub.Module(name)


default_module_manager = ModuleManager()
