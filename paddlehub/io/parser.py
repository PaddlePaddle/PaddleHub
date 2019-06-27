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

import codecs
import sys
import yaml

from paddlehub.common.utils import sys_stdin_encoding


class CSVFileParser(object):
    def __init__(self):
        pass

    def _check(self):
        pass

    def parse(self, csv_file):
        with codecs.open(csv_file, "r", sys_stdin_encoding()) as file:
            content = file.read()
        content = content.split('\n')
        self.title = content[0].split(',')
        self.content = {}
        for key in self.title:
            self.content[key] = []

        for text in content[1:]:
            if (text == ""):
                continue

            for index, item in enumerate(text.split(',')):
                title = self.title[index]
                self.content[title].append(item)

        return self.content


class YAMLFileParser(object):
    def __init__(self):
        pass

    def _check(self):
        pass

    def parse(self, yaml_file):
        with codecs.open(yaml_file, "r", sys_stdin_encoding()) as file:
            content = file.read()
        return yaml.load(content, Loader=yaml.BaseLoader)


class TextFileParser(object):
    def __init__(self):
        pass

    def _check(self):
        pass

    def parse(self, txt_file):
        with codecs.open(txt_file, "r", sys_stdin_encoding()) as file:
            contents = []
            for line in file:
                line = line.strip()
                if line:
                    contents.append(line)
        return contents


csv_parser = CSVFileParser()
yaml_parser = YAMLFileParser()
txt_parser = TextFileParser()
