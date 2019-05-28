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

from enum import Enum

from PIL import Image

from paddlehub.common.logger import logger
from paddlehub.common import utils


class DataType(Enum):
    IMAGE = 0
    TEXT = 1
    AUDIO = 2
    VIDEO = 3
    INT = 4
    FLOAT = 5

    @classmethod
    def type(cls, data_type):
        if data_type in DataType:
            return data_type
        data_type = data_type.upper()
        if data_type in DataType.__dict__:
            return DataType.__dict__[data_type]
        return None

    @classmethod
    def str(cls, data_type):
        if data_type == DataType.IMAGE:
            return "IMAGE"
        elif data_type == DataType.TEXT:
            return "TEXT"
        elif data_type == DataType.AUDIO:
            return "AUDIO"
        elif data_type == DataType.VIDEO:
            return "VIDEO"
        elif data_type == DataType.INT:
            return "INT"
        elif data_type == DataType.FLOAT:
            return "FLOAT"
        return None

    @classmethod
    def is_valid_type(cls, data_type):
        data_type = DataType.type(data_type)
        return data_type in DataType
