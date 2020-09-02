# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import functools
import logging
import sys
import time
from typing import List

import colorlog
from colorama import Fore

log_config = {
    'DEBUG': {
        'level': 10,
        'color': 'purple'
    },
    'INFO': {
        'level': 20,
        'color': 'green'
    },
    'TRAIN': {
        'level': 21,
        'color': 'cyan'
    },
    'EVAL': {
        'level': 22,
        'color': 'blue'
    },
    'WARNING': {
        'level': 30,
        'color': 'yellow'
    },
    'ERROR': {
        'level': 40,
        'color': 'red'
    },
    'CRITICAL': {
        'level': 50,
        'color': 'bold_red'
    }
}


class Logger(object):
    def __init__(self, name=None):
        name = 'PaddleHub' if not name else name
        self.logger = logging.getLogger(name)

        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(self.logger.log, conf['level'])
            self.__dict__[key.lower()] = functools.partial(self.logger.log, conf['level'])

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] [%(levelname)8s] - %(message)s',
            log_colors={key: conf['color']
                        for key, conf in log_config.items()})

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = "DEBUG"
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False


class ProgressBar(object):
    '''
    '''

    def __init__(self, title: str, flush_interval: float = 0.1):
        self.last_flush_time = time.time()
        self.flush_interval = flush_interval
        self._end = False
        self.title = title

    def __enter__(self):
        sys.stdout.write('{}\n'.format(self.title))
        return self

    def __exit__(self, exit_exception, exit_value, exit_traceback):
        if not exit_value:
            self._end = True
            self.update(1)
        else:
            sys.stdout.write('\n')

    def update(self, progress):
        msg = '[{:<50}] {:.2f}%'.format('#' * int(progress * 50), progress * 100)
        need_flush = (time.time() - self.last_flush_time) >= self.flush_interval

        if need_flush or self._end:
            sys.stdout.write('\r{}'.format(msg))
            self.last_flush_time = time.time()
            sys.stdout.flush()

        if self._end:
            sys.stdout.write('\n')


class _FormattedText(object):
    _MAP = {'red': Fore.RED, 'yellow': Fore.YELLOW, 'green': Fore.GREEN, 'blue': Fore.BLUE}

    def __init__(self, text: str, width: int, align='<', color=None):
        self.text = text
        self.align = align
        self.color = _FormattedText._MAP[color] if color else color
        self.width = width

    def __repr__(self):
        form = ':{}{}'.format(self.align, self.width)
        text = ('{' + form + '}').format(self.text)
        if not self.color:
            return text
        return self.color + text + Fore.RESET


class TableCell(object):
    def __init__(self, content: str = '', width: int = 0, align: str = '<', color: str = ''):
        self._width = width if width else len(content)
        self._width = 1 if self._width < 1 else self._width
        self._contents = []
        for i in range(0, len(content), self._width):
            text = _FormattedText(content[i:i + self._width], width, align, color)
            self._contents.append(text)
        self.align = align
        self.color = color

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value
        for content in self._contents:
            content.width = value

    @property
    def height(self) -> int:
        return len(self._contents)

    @height.setter
    def height(self, value: int):
        if value < self.height:
            raise RuntimeError(self.height, value)
        self._contents += [_FormattedText('', width=self.width, align=self.align, color=self.color)
                           ] * (value - self.height)

    def __len__(self) -> int:
        return len(self._contents)

    def __getitem__(self, idx: int) -> str:
        return self._contents[idx]

    def __repr__(self) -> str:
        return '\n'.join([str(item) for item in self._contents])


class TableLine(object):
    def __init__(self):
        self.cells = []

    def append(self, cell: TableCell):
        self.cells.append(cell)

    @property
    def width(self):
        _width = 0
        for cell in self.cells():
            _width += cell.width
        return _width

    @property
    def height(self):
        _height = -1
        for cell in self.cells:
            _height = max(_height, cell.height)
        return _height

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        content = ''
        for i in range(self.height):
            content += '|'
            for cell in self.cells:
                if i > cell.height:
                    content = content + '|'
                else:
                    content = content + str(cell[i]) + '|'
            content += '\n'
        return content

    def __getitem__(self, idx: int) -> TableCell:
        return self.cells[idx]


class TableRow(object):
    def __init__(self):
        self.cells = []

    def append(self, cell: TableCell):
        self.cells.append(cell)

    @property
    def width(self):
        _width = -1
        for cell in self.cells:
            _width = max(_width, cell.width)
        return _width

    @property
    def height(self):
        _height = 0
        for cell in self.cells:
            _height += cell.height
        return _height

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx: int) -> TableCell:
        return self.cells[idx]


class Table(object):
    def __init__(self):
        self.lines = []
        self.rows = []

    def append(self, *contents, colors: List[str] = [], aligns: List[str] = [], widths: List[int] = []):
        newline = TableLine()

        for idx, content in enumerate(contents):
            width = widths[idx] if idx < len(widths) else len(content)
            color = colors[idx] if idx < len(colors) else ''
            align = aligns[idx] if idx < len(aligns) else ''

            newcell = TableCell(content, width=width, color=color, align=align)
            newline.append(newcell)
            if idx >= len(self.rows):
                newrow = TableRow()

                for line in self.lines:
                    cell = TableCell(width=width, color=color, align=align)
                    line.append(cell)
                    newrow.append(cell)
                newrow.append(newcell)
                self.rows.append(newrow)
            else:
                self.rows[idx].append(newcell)

        for idx in range(len(newline), len(self.rows)):
            width = widths[idx] if idx < len(widths) else self.rows[idx].width
            color = colors[idx] if idx < len(colors) else ''
            align = aligns[idx] if idx < len(aligns) else ''
            cell = TableCell(width=width, color=color, align=align)
            newline.append(cell)

        self.lines.append(newline)
        self._adjust()

    def _adjust(self):
        for row in self.rows:
            _width = -1
            for cell in row:
                _width = max(_width, cell.width)
            for cell in row:
                cell.width = _width

        for line in self.lines:
            _height = -1
            for cell in line:
                _height = max(_height, cell.height)
            for cell in line:
                cell.height = _height

    @property
    def width(self):
        _width = -1
        for line in self.lines:
            _width = max(_width, line.width)
        return _width

    @property
    def height(self):
        _height = -1
        for row in self.rows:
            _height = max(_height, row.height)
        return _height

    def __repr__(self):
        sepline = '+{}+\n'.format('+'.join(['-' * row.width for row in self.rows]))
        content = ''
        for line in self.lines:
            content = content + str(line)
            content += sepline
        return sepline + content

logger = Logger()
