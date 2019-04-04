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

color_dict = {
    "white": "\033[1;37m%s\033[0m",
    "black": "\033[30m%s\033[0m",
    "dark_gray": "\033[1;30m%s\033[0m",
    "light_gray": "\033[0;37m%s\033[0m",
    "blue": "\033[0;34m%s\033[0m",
    "light_blue": "\033[1;34m%s\033[0m",
    "green": "\033[0;32m%s\033[0m",
    "light_green": "\033[1;32m%s\033[0m",
    "cyan": "\033[0;36m%s\033[0m",
    "light_cyan": "\033[1;36m%s\033[0m",
    "red": "\033[0;31m%s\033[0m",
    "light_red": "\033[1;31m%s\033[0m",
    "purple": "\033[0;35m%s\033[0m",
    "light_purple": "\033[1;35m%s\033[0m",
    "brown": "\033[0;33m%s\033[0m",
    "yellow": "\033[1;33m%s\033[0m"
}


def colorful_text(color, text):
    if color not in color_dict:
        color = color_dict['blue']
    else:
        color = color_dict[color]
    return color % text


class TablePrinter:
    def __init__(self, titles, placeholders):
        self.titles = titles
        self.placeholders = placeholders
        self.text = "\n"
        self.add_title()

    def add_horizontal_line(self):
        line = '+'
        for value in self.placeholders:
            line += '-' * (value + 2) + '+'
        line += '\n'
        self.text += line

    def add_title(self):
        self.add_horizontal_line()
        title_text = "|"
        for index, title in enumerate(self.titles):
            title = colorful_text("light_green", title)
            _ph = 11
            title_text += (
                "{0:^%d}|" % (self.placeholders[index] + 2 + _ph)).format(title)
        title_text += '\n'
        self.text += title_text

    def add_line(self, contents, colors=None):
        self.add_horizontal_line()
        max_lines = 0
        marks = [False] * len(contents)
        colors = [None] * len(contents) if colors is None else colors
        offset = [0] * len(contents)
        for index, content in enumerate(contents):
            content_length = int(len(content) / self.placeholders[index])
            if content_length > 0:
                marks[index] = True
            if content_length > max_lines:
                max_lines = content_length

        line = ''
        for cnt in range(max_lines + 1):
            line += '|'
            for index, content in enumerate(contents):
                length = self.placeholders[index]
                split_text = content[offset[index]:offset[index] + length]
                if colors[index] and split_text:
                    split_text = colorful_text(colors[index], split_text)
                    _ph = 11
                else:
                    _ph = 0
                align = "<" if marks[index] else "^"
                line += (
                    "{0:%s%d}|" % (align, self.placeholders[index] + 2 + _ph)
                ).format(split_text)
                offset[index] += length
            line += '\n'

        self.text += line

    def get_text(self):
        self.add_horizontal_line()
        return self.text
