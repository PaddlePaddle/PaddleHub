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

from enum import Enum
from PIL import Image
from paddle_hub.tools.logger import logger
from paddle_hub.tools import utils


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

    @classmethod
    def type_reader(cls, data_type):
        data_type = DataType.type(data_type)
        if not DataType.is_valid_type(data_type):
            logger.critical("invalid data type %s" % data_type)
            exit(1)

        if data_type == DataType.IMAGE:
            return ImageReader
        elif data_type == DataType.TEXT:
            return TextReader
        else:
            type_str = DataType.str(data_type)
            logger.critical(
                "data type %s not supported for the time being" % type_str)
            exit(1)


class ImageReader:
    @classmethod
    def read(cls, path):
        utils.check_path(path)
        image = Image.open(path)
        return image


class TextReader:
    @classmethod
    def read(cls, text):
        return text
