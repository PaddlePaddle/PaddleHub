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

from __future__ import print_function
from __future__ import division
from __future__ import print_function

import logging
import math


class Logger(object):
    PLACEHOLDER = '%'
    NOLOG = "NOLOG"

    def __init__(self, name=None):
        logging.basicConfig(
            format='[%(asctime)-15s] [%(levelname)8s] - %(message)s')

        if not name:
            name = "PaddleHub"

        self.logger = logging.getLogger(name)
        self.logLevel = "DEBUG"
        self.logger.setLevel(self._get_logging_level())

    def _is_no_log(self):
        return self.getLevel() == Logger.NOLOG

    def _get_logging_level(self):
        return eval("logging.%s" % self.logLevel)

    def setLevel(self, logLevel):
        self.logLevel = logLevel.upper()
        if not self._is_no_log():
            _logging_level = eval("logging.%s" % self.logLevel)
            self.logger.setLevel(_logging_level)

    def getLevel(self):
        return self.logLevel

    def __call__(self, type, msg):
        def _get_log_arr(msg, len_limit=30):
            ph = Logger.PLACEHOLDER
            lrspace = 2
            lc = rc = " " * lrspace
            tbspace = 1
            msgarr = str(msg).split("\n")
            if len(msgarr) == 1:
                return msgarr

            temp_arr = msgarr
            msgarr = []
            for text in temp_arr:
                if len(text) > len_limit:
                    for i in range(math.ceil(len(text) / len_limit)):
                        if i == 0:
                            msgarr.append(text[0:len_limit])
                        else:
                            fr = len_limit + (len_limit - 4) * (i - 1)
                            to = len_limit + (len_limit - 4) * i
                            if to > len(text):
                                to = len(text)
                            msgarr.append("===>" + text[fr:to])
                else:
                    msgarr.append(text)

            maxlen = -1
            for text in msgarr:
                if len(text) > maxlen:
                    maxlen = len(text)

            result = [" ", ph * (maxlen + 2 + lrspace * 2)]
            tbline = "%s%s%s" % (ph, " " * (maxlen + lrspace * 2), ph)
            for index in range(tbspace):
                result.append(tbline)
            for text in msgarr:
                text = "%s%s%s%s%s%s" % (ph, lc, text, rc, " " *
                                         (maxlen - len(text)), ph)
                result.append(text)
            for index in range(tbspace):
                result.append(tbline)
            result.append(ph * (maxlen + 2 + lrspace * 2))
            return result

        if self._is_no_log():
            return

        func = eval("self.logger.%s" % type)
        for msg in _get_log_arr(msg):
            func(msg)

    def debug(self, msg):
        self("debug", msg)

    def info(self, msg):
        self("info", msg)

    def error(self, msg):
        self("error", msg)

    def warning(self, msg):
        self("warning", msg)

    def critical(self, msg):
        self("critical", msg)


logger = Logger()
