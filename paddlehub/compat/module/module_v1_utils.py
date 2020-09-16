# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from easydict import EasyDict

from paddlehub.compat.module import module_desc_pb2


def convert_module_desc(desc_file):
    desc = module_desc_pb2.ModuleDesc()
    with open(desc_file, 'rb') as file:
        desc.ParseFromString(file.read())

    result = convert_attr(desc.attr)
    result.signatures = convert_signatures(desc.sign2var)
    return result


def convert_signatures(signmaps):
    _dict = EasyDict()
    for sign, var in signmaps.items():
        _dict[sign] = EasyDict(inputs=[], outputs=[])
        for fetch_var in var.fetch_desc:
            _dict[sign].outputs.append(EasyDict(name=fetch_var.var_name, alias=fetch_var.alias))

        for feed_var in var.feed_desc:
            _dict[sign].inputs.append(EasyDict(name=feed_var.var_name, alias=feed_var.alias))

    return _dict


def convert_attr(module_attr):
    if module_attr.type == 1:
        return module_attr.i
    elif module_attr.type == 2:
        return module_attr.f
    elif module_attr.type == 3:
        return module_attr.s
    elif module_attr.type == 4:
        return module_attr.b

    _dict = EasyDict()
    for key, val in module_attr.map.data.items():
        _dict[key] = convert_attr(val)
    return _dict
