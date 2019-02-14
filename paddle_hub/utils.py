#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from paddle_hub import module_desc_pb2
from paddle_hub.logger import logger
import paddle
import paddle.fluid as fluid
import os


def to_list(input):
    if not isinstance(input, list):
        if not isinstance(input, tuple):
            input = [input]

    return input


def mkdir(path):
    """ the same as the shell command mkdir -p "
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_keyed_type_of_pyobj(pyobj):
    if isinstance(pyobj, bool):
        return module_desc_pb2.BOOLEAN
    elif isinstance(pyobj, int):
        return module_desc_pb2.INT
    elif isinstance(pyobj, str):
        return module_desc_pb2.STRING
    elif isinstance(pyobj, float):
        return module_desc_pb2.FLOAT
    return module_desc_pb2.STRING


def get_pykey(key, keyed_type):
    if keyed_type == module_desc_pb2.BOOLEAN:
        return bool(key)
    elif keyed_type == module_desc_pb2.INT:
        return int(key)
    elif keyed_type == module_desc_pb2.STRING:
        return str(key)
    elif keyed_type == module_desc_pb2.FLOAT:
        return float(key)
    return str(key)


#TODO(wuzewu): solving the problem of circular references
def from_pyobj_to_flexible_data(pyobj, flexible_data, obj_filter=None):
    if obj_filter and obj_filter(pyobj):
        logger.info("filter python object")
        return
    if isinstance(pyobj, bool):
        flexible_data.type = module_desc_pb2.BOOLEAN
        flexible_data.b = pyobj
    elif isinstance(pyobj, int):
        flexible_data.type = module_desc_pb2.INT
        flexible_data.i = pyobj
    elif isinstance(pyobj, str):
        flexible_data.type = module_desc_pb2.STRING
        flexible_data.s = pyobj
    elif isinstance(pyobj, float):
        flexible_data.type = module_desc_pb2.FLOAT
        flexible_data.f = pyobj
    elif isinstance(pyobj, list) or isinstance(pyobj, tuple):
        flexible_data.type = module_desc_pb2.LIST
        for index, obj in enumerate(pyobj):
            from_pyobj_to_flexible_data(
                obj, flexible_data.list.data[str(index)], obj_filter)
    elif isinstance(pyobj, set):
        flexible_data.type = module_desc_pb2.SET
        for index, obj in enumerate(list(pyobj)):
            from_pyobj_to_flexible_data(obj, flexible_data.set.data[str(index)],
                                        obj_filter)
    elif isinstance(pyobj, dict):
        flexible_data.type = module_desc_pb2.MAP
        for key, value in pyobj.items():
            from_pyobj_to_flexible_data(value, flexible_data.map.data[str(key)],
                                        obj_filter)
            flexible_data.map.keyType[str(key)] = get_keyed_type_of_pyobj(key)
    elif isinstance(pyobj, type(None)):
        flexible_data.type = module_desc_pb2.NONE
    else:
        flexible_data.type = module_desc_pb2.OBJECT
        flexible_data.name = str(pyobj.__class__.__name__)
        if not hasattr(pyobj, "__dict__"):
            logger.warning(
                "python obj %s has not __dict__ attr" % flexible_data.name)
            return
        for key, value in pyobj.__dict__.items():
            from_pyobj_to_flexible_data(
                value, flexible_data.object.data[str(key)], obj_filter)
            flexible_data.object.keyType[str(key)] = get_keyed_type_of_pyobj(
                key)


def from_flexible_data_to_pyobj(flexible_data):
    if flexible_data.type == module_desc_pb2.BOOLEAN:
        result = flexible_data.b
    elif flexible_data.type == module_desc_pb2.INT:
        result = flexible_data.i
    elif flexible_data.type == module_desc_pb2.STRING:
        result = flexible_data.s
    elif flexible_data.type == module_desc_pb2.FLOAT:
        result = flexible_data.f
    elif flexible_data.type == module_desc_pb2.LIST:
        result = []
        for index in range(len(flexible_data.list.data)):
            result.append(
                from_flexible_data_to_pyobj(flexible_data.m.data(str(index))))
    elif flexible_data.type == module_desc_pb2.SET:
        result = set()
        for index in range(len(flexible_data.set.data)):
            result.add(
                from_flexible_data_to_pyobj(flexible_data.m.data(str(index))))
    elif flexible_data.type == module_desc_pb2.MAP:
        result = {}
        for key, value in flexible_data.map.data.items():
            key = get_pykey(key, flexible_data.map.keyType[key])
            result[key] = from_flexible_data_to_pyobj(value)
    elif flexible_data.type == module_desc_pb2.NONE:
        result = None
    elif flexible_data.type == module_desc_pb2.OBJECT:
        result = None
        logger.warning("can't tran flexible_data to python object")
    else:
        result = None
        logger.warning("unknown type of flexible_data")

    return result
