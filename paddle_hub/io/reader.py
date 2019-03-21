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

import yaml


class CSVReader:
    def __init__(self):
        pass

    def _check(self):
        pass

    def read(self, csv_file):
        with open(csv_file, "r") as file:
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


class YAMLReader:
    def __init__(self):
        pass

    def _check(self):
        pass

    def read(self, yaml_file):
        with open(yaml_file, "r") as file:
            content = file.read()
        return yaml.load(content)


yaml_reader = YAMLReader()
csv_reader = CSVReader()
