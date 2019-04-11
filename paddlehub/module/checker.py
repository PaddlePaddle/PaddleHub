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

import paddle

from paddlehub.common.logger import logger
from paddlehub.module import check_info_pb2
from paddlehub.version import hub_version, module_proto_version

# check info
CHECK_INFO_PB_FILENAME = "check_info.pb"
FILE_SEP = "/"


class ModuleChecker(object):
    def __init__(self, module_path):
        self.module_path = module_path

    def generate_check_info(self):
        check_info = check_info_pb2.CheckInfo()
        check_info.paddle_version = paddle.__version__
        check_info.hub_version = hub_version
        check_info.module_proto_version = module_proto_version
        file_infos = check_info.file_infos
        file_list = [file for file in os.listdir(self.module_path)]
        while file_list:
            file = file_list[0]
            file_list = file_list[1:]
            abs_path = os.path.join(self.module_path, file)
            if os.path.isdir(abs_path):
                for sub_file in os.listdir(abs_path):
                    sub_file = os.path.join(file, sub_file)
                    file_list.append(sub_file)
                file_info = file_infos.add()
                file_info.file_name = file
                file.replace(os.sep, FILE_SEP)
                file_info.type = check_info_pb2.DIR
                file_info.is_need = True
            else:
                file.replace(os.sep, FILE_SEP)
                file_info = file_infos.add()
                file_info.file_name = file
                file_info.type = check_info_pb2.FILE
                file_info.is_need = True

        with open(os.path.join(self.module_path, CHECK_INFO_PB_FILENAME),
                  "wb") as fi:
            fi.write(check_info.SerializeToString())

    @property
    def module_proto_version(self):
        return self.check_info.module_proto_version

    @property
    def hub_version(self):
        return self.check_info.hub_version

    @property
    def paddle_version(self):
        return self.check_info.paddle_version

    @property
    def file_infos(self):
        return self.check_info.file_infos

    def check(self):
        self.check_info_pb_path = os.path.join(self.module_path,
                                               CHECK_INFO_PB_FILENAME)

        if not (os.path.exists(self.check_info_pb_path)
                or os.path.isfile(self.check_info_pb_path)):
            logger.error("this module lack of key documents [%s]" %
                         CHECK_INFO_PB_FILENAME)
            return False

        self.check_info = check_info_pb2.CheckInfo()
        try:
            with open(self.check_info_pb_path, "rb") as fi:
                pb_string = fi.read()
                result = self.check_info.ParseFromString(pb_string)
                if len(pb_string) == 0 or (result is not None
                                           and result != len(pb_string)):
                    logger.error(
                        "the [%s] file is incomplete" % CHECK_INFO_PB_FILENAME)
                    return False
        except Exception as e:
            return False

        if not self.check_info.paddle_version:
            logger.error(
                "can't read paddle version from [%s]" % CHECK_INFO_PB_FILENAME)
            return False

        if not self.check_info.hub_version:
            logger.error(
                "can't read hub version from [%s]" % CHECK_INFO_PB_FILENAME)
            return False

        if not self.check_info.module_proto_version:
            logger.error("can't read module pb version from [%s]" %
                         CHECK_INFO_PB_FILENAME)
            return False

        if not self.check_info.file_infos:
            logger.error(
                "can't read file info from [%s]" % CHECK_INFO_PB_FILENAME)
            return False

        return self.check_module() and self.check_compatibility()

    def check_compatibility(self):
        return self._check_module_proto_version() and self._check_hub_version(
        ) and self._check_paddle_version()

    def check_module(self):
        return self._check_module_integrity() and self._check_dependency()

    def _check_dependency(self):
        return True

    def _check_module_proto_version(self):
        if self.module_proto_version != module_proto_version:
            return False
        return True

    def _check_hub_version(self):
        return True

    def _check_paddle_version(self):
        return True

    def _check_module_integrity(self):

        for file_info in self.file_infos:
            file_type = file_info.type
            file_path = file_info.file_name.replace(FILE_SEP, os.sep)
            file_path = os.path.join(self.module_path, file_path)
            if not os.path.exists(file_path):
                if file_info.is_need:
                    logger.error(
                        "Module incompleted! Missing file [%s]" % file_path)
                    return False
            else:
                if file_type == check_info_pb2.FILE:
                    if not os.path.isfile(file_path):
                        logger.error("File type error %s" % file_path)
                        return False

                if file_type == check_info_pb2.DIR:
                    if not os.path.isdir(file_path):
                        logger.error("File type error %s" % file_path)
                        return False
        return True
