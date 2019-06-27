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

import sys
import os
import time
import multiprocessing
import hashlib
import platform

import paddle
import paddle.fluid as fluid
import six

from paddlehub.module import module_desc_pb2
from paddlehub.common.logger import logger


def version_compare(version1, version2):
    version1 = version1.split(".")
    version2 = version2.split(".")
    num = min(len(version1), len(version2))
    for index in range(num):
        try:
            vn1 = int(version1[index])
        except:
            vn1 = 0
        try:
            vn2 = int(version2[index])
        except:
            vn2 = 0

        if vn1 > vn2:
            return True
        elif vn1 < vn2:
            return False
    return len(version1) > len(version2)


def get_platform():
    return platform.platform()


def is_windows():
    return get_platform().lower().startswith("windows")


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


def md5_of_file(file):
    md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    return md5.hexdigest()


def md5(text):
    if isinstance(text, str):
        text = text.encode("utf8")
    md5 = hashlib.md5()
    md5.update(text)
    return md5.hexdigest()


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
        return key == "True"
    elif keyed_type == module_desc_pb2.INT:
        return int(key)
    elif keyed_type == module_desc_pb2.STRING:
        return str(key)
    elif keyed_type == module_desc_pb2.FLOAT:
        return float(key)
    return str(key)


#TODO(wuzewu): solving the problem of circular references
def from_pyobj_to_module_attr(pyobj, module_attr, obj_filter=None):
    if obj_filter and obj_filter(pyobj):
        return
    if isinstance(pyobj, bool):
        module_attr.type = module_desc_pb2.BOOLEAN
        module_attr.b = pyobj
    elif isinstance(pyobj, six.integer_types):
        module_attr.type = module_desc_pb2.INT
        module_attr.i = pyobj
    elif isinstance(pyobj, six.text_type):
        module_attr.type = module_desc_pb2.STRING
        module_attr.s = pyobj
    elif isinstance(pyobj, six.binary_type):
        module_attr.type = module_desc_pb2.STRING
        module_attr.s = pyobj
    elif isinstance(pyobj, float):
        module_attr.type = module_desc_pb2.FLOAT
        module_attr.f = pyobj
    elif isinstance(pyobj, list) or isinstance(pyobj, tuple):
        module_attr.type = module_desc_pb2.LIST
        for index, obj in enumerate(pyobj):
            from_pyobj_to_module_attr(obj, module_attr.list.data[str(index)],
                                      obj_filter)
    elif isinstance(pyobj, set):
        module_attr.type = module_desc_pb2.SET
        for index, obj in enumerate(list(pyobj)):
            from_pyobj_to_module_attr(obj, module_attr.set.data[str(index)],
                                      obj_filter)
    elif isinstance(pyobj, dict):
        module_attr.type = module_desc_pb2.MAP
        for key, value in pyobj.items():
            from_pyobj_to_module_attr(value, module_attr.map.data[str(key)],
                                      obj_filter)
            module_attr.map.key_type[str(key)] = get_keyed_type_of_pyobj(key)
    elif isinstance(pyobj, type(None)):
        module_attr.type = module_desc_pb2.NONE
    else:
        module_attr.type = module_desc_pb2.OBJECT
        module_attr.name = str(pyobj.__class__.__name__)
        if not hasattr(pyobj, "__dict__"):
            logger.warning(
                "python obj %s has not __dict__ attr" % module_attr.name)
            return
        for key, value in pyobj.__dict__.items():
            from_pyobj_to_module_attr(value, module_attr.object.data[str(key)],
                                      obj_filter)
            module_attr.object.key_type[str(key)] = get_keyed_type_of_pyobj(key)


def from_module_attr_to_pyobj(module_attr):
    if module_attr.type == module_desc_pb2.BOOLEAN:
        result = module_attr.b
    elif module_attr.type == module_desc_pb2.INT:
        result = module_attr.i
    elif module_attr.type == module_desc_pb2.STRING:
        result = module_attr.s
    elif module_attr.type == module_desc_pb2.FLOAT:
        result = module_attr.f
    elif module_attr.type == module_desc_pb2.LIST:
        result = []
        for index in range(len(module_attr.list.data)):
            result.append(
                from_module_attr_to_pyobj(module_attr.list.data[str(index)]))
    elif module_attr.type == module_desc_pb2.SET:
        result = set()
        for index in range(len(module_attr.set.data)):
            result.add(
                from_module_attr_to_pyobj(module_attr.set.data[str(index)]))
    elif module_attr.type == module_desc_pb2.MAP:
        result = {}
        for key, value in module_attr.map.data.items():
            key = get_pykey(key, module_attr.map.key_type[key])
            result[key] = from_module_attr_to_pyobj(value)
    elif module_attr.type == module_desc_pb2.NONE:
        result = None
    elif module_attr.type == module_desc_pb2.OBJECT:
        result = None
        logger.warning("can't tran module attr to python object")
    else:
        result = None
        logger.warning("unknown type of module attr")

    return result


def check_path(path):
    pass


def check_url(url):
    pass


def get_file_ext(file_path):
    return os.path.splitext(file_path)[-1]


def is_csv_file(file_path):
    return get_file_ext(file_path) == ".csv"


def is_yaml_file(file_path):
    return get_file_ext(file_path) == ".yml"


def get_running_device_info(config):
    if config.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    return place, dev_count


def get_platform_default_encoding():
    if platform.platform().lower().startswith("windows"):
        return "gbk"
    return "utf8"


def sys_stdin_encoding():
    encoding = sys.stdin.encoding
    if encoding is None:
        encoding = sys.getdefaultencoding()

    if encoding is None:
        encoding = get_platform_default_encoding()
    return encoding


def sys_stdout_encoding():
    encoding = sys.stdout.encoding
    if encoding is None:
        encoding = sys.getdefaultencoding()

    if encoding is None:
        encoding = get_platform_default_encoding()
    return encoding
