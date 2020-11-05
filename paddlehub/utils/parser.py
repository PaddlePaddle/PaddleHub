# coding:utf-8
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

import codecs
from typing import List

import yaml

from paddlehub.utils.utils import sys_stdin_encoding


class CSVFileParser(object):
    def parse(self, csv_file: str) -> dict:
        with codecs.open(csv_file, 'r', sys_stdin_encoding()) as file:
            content = file.read()
        content = content.split('\n')
        self.title = content[0].split(',')
        self.content = {}
        for key in self.title:
            self.content[key] = []

        for text in content[1:]:
            if (text == ''):
                continue

            for index, item in enumerate(text.split(',')):
                title = self.title[index]
                self.content[title].append(item)

        return self.content


class YAMLFileParser(object):
    def parse(self, yaml_file: str) -> dict:
        with codecs.open(yaml_file, 'r', sys_stdin_encoding()) as file:
            content = file.read()
        return yaml.load(content, Loader=yaml.BaseLoader)


class TextFileParser(object):
    def parse(self, txt_file: str, use_strip: bool = True) -> List:
        contents = []
        try:
            with codecs.open(txt_file, 'r', encoding='utf8') as file:
                for line in file:
                    if use_strip:
                        line = line.strip()
                    if line:
                        contents.append(line)
        except:
            with codecs.open(txt_file, 'r', encoding='gbk') as file:
                for line in file:
                    if use_strip:
                        line = line.strip()
                    if line:
                        contents.append(line)
        return contents


csv_parser = CSVFileParser()
yaml_parser = YAMLFileParser()
txt_parser = TextFileParser()
