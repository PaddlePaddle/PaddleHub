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

import contextlib
import copy
import functools
import logging
import os
import sys
import time
import threading
from typing import List

import colorlog
from colorama import Fore

import paddlehub.config as hubconf
from paddlehub.env import LOG_HOME

loggers = {}

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
    '''
    Deafult logger in PaddleHub

    Args:
        name(str) : Logger name, default is 'PaddleHub'
    '''

    def __init__(self, name: str = None):
        name = 'PaddleHub' if not name else name
        self.logger = logging.getLogger(name)

        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(self.__call__, conf['level'])
            self.__dict__[key.lower()] = functools.partial(self.__call__, conf['level'])

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s',
            log_colors={key: conf['color']
                        for key, conf in log_config.items()})

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = hubconf.log_level
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._is_enable = hubconf.log_enable

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: str, msg: str):
        if not self.is_enable:
            return

        self.logger.log(log_level, msg)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        '''
        Continuously print a progress bar with rotating special effects.

        Args:
            msg(str): Message to be printed.
            interval(float): Rotation interval. Default to 0.1.
        '''
        end = False

        def _printer():
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.info('{}: {}'.format(msg, flag))
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.daemon = True
        t.start()
        yield
        end = True


class ProgressBar(object):
    '''
    Progress bar printer

    Args:
        title(str) : Title text
        flush_interval(float): Flush rate of progress bar, default is 0.1.

    Examples:
        .. code-block:: python

            with ProgressBar('Download module') as bar:
                for i in range(100):
                    bar.update(i / 100)

            # with continuous bar.update, the progress bar in the terminal
            # will continue to update until 100%
            #
            # Download module
            # [##################################################] 100.00%
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

    def update(self, progress: float):
        '''
        Update progress bar

        Args:
            progress: Processing progress, from 0.0 to 1.0
        '''
        msg = '[{:<50}] {:.2f}%'.format('#' * int(progress * 50), progress * 100)
        need_flush = (time.time() - self.last_flush_time) >= self.flush_interval

        if need_flush or self._end:
            sys.stdout.write('\r{}'.format(msg))
            self.last_flush_time = time.time()
            sys.stdout.flush()

        if self._end:
            sys.stdout.write('\n')


class FormattedText(object):
    '''
    Cross-platform formatted string

    Args:
        text(str) : Text content
        width(int) : Text length, if the text is less than the specified length, it will be filled with spaces
        align(str) : Text alignment, it must be:
            ========   ====================================
            Charater   Meaning
            --------   ------------------------------------
            '<'        The text will remain left aligned
            '^'        The text will remain middle aligned
            '>'        The text will remain right aligned
            ========   ====================================
        color(str) : Text color, default is None(depends on terminal configuration)
    '''
    _MAP = {'red': Fore.RED, 'yellow': Fore.YELLOW, 'green': Fore.GREEN, 'blue': Fore.BLUE, 'cyan': Fore.CYAN}

    def __init__(self, text: str, width: int = None, align: str = '<', color: str = None):
        self.text = text
        self.align = align
        self.color = FormattedText._MAP[color] if color else color
        self.width = width if width else len(self.text)

    def __repr__(self) -> str:
        form = '{{:{}{}}}'.format(self.align, self.width)
        text = form.format(self.text)
        if not self.color:
            return text
        return self.color + text + Fore.RESET


class TableCell(object):
    '''The basic components of a table'''

    def __init__(self, content: str = '', width: int = 0, align: str = '<', color: str = ''):
        self._width = width if width else len(content)
        self._width = 1 if self._width < 1 else self._width
        self._contents = []
        for i in range(0, len(content), self._width):
            text = FormattedText(content[i:i + self._width], width, align, color)
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
        self._contents += [FormattedText('', width=self.width, align=self.align, color=self.color)
                           ] * (value - self.height)

    def __len__(self) -> int:
        return len(self._contents)

    def __getitem__(self, idx: int) -> str:
        return self._contents[idx]

    def __repr__(self) -> str:
        return '\n'.join([str(item) for item in self._contents])


class TableRow(object):
    '''Table row composed of TableCell'''

    def __init__(self):
        self.cells = []

    def append(self, cell: TableCell):
        self.cells.append(cell)

    @property
    def width(self) -> int:
        _width = 0
        for cell in self.cells():
            _width += cell.width
        return _width

    @property
    def height(self) -> int:
        _height = -1
        for cell in self.cells:
            _height = max(_height, cell.height)
        return _height

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
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


class TableColumn(object):
    '''Table column composed of TableCell'''

    def __init__(self):
        self.cells = []

    def append(self, cell: TableCell):
        self.cells.append(cell)

    @property
    def width(self) -> int:
        _width = -1
        for cell in self.cells:
            _width = max(_width, cell.width)
        return _width

    @property
    def height(self) -> int:
        _height = 0
        for cell in self.cells:
            _height += cell.height
        return _height

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> TableCell:
        return self.cells[idx]


class Table(object):
    '''
    Table with adaptive width and height

    Args:
        colors(list[str]) : Text colors
        aligns(list[str]) : Text alignments
        widths(list[str]) : Text widths

    Examples:
        .. code-block:: python

            table = Table(widths=[12, 20])
            table.append('name', 'PaddleHub')
            table.append('version', '2.0.0')
            table.append(
                'description',
                'PaddleHub is a pretrainied model application tool under the PaddlePaddle')
            table.append('author')

            print(table)

            # the result is
            # +------------+--------------------+
            # |name        |PaddleHub           |
            # +------------+--------------------+
            # |version     |2.0.0               |
            # +------------+--------------------+
            # |description |PaddleHub is a pretr|
            # |            |ainied model applica|
            # |            |tion tool under the |
            # |            |PaddlePaddle        |
            # +------------+--------------------+
            # |author      |                    |
            # +------------+--------------------+
    '''

    def __init__(self, colors: List[str] = [], aligns: List[str] = [], widths: List[int] = []):
        self.rows = []
        self.columns = []
        self.colors = colors
        self.aligns = aligns
        self.widths = widths

    def append(self, *contents, colors: List[str] = [], aligns: List[str] = [], widths: List[int] = []):
        '''
        Add a row to the table

        Args:
            *contents(*list): Contents of the row, each content will be placed in a separate cell
            colors(list[str]) : Text colors
            aligns(list[str]) : Text alignments
            widths(list[str]) : Text widths
        '''
        newrow = TableRow()

        widths = copy.deepcopy(self.widths) if not widths else widths
        colors = copy.deepcopy(self.colors) if not colors else colors
        aligns = copy.deepcopy(self.aligns) if not aligns else aligns

        for idx, content in enumerate(contents):
            width = widths[idx] if idx < len(widths) else len(content)
            color = colors[idx] if idx < len(colors) else ''
            align = aligns[idx] if idx < len(aligns) else ''

            newcell = TableCell(content, width=width, color=color, align=align)
            newrow.append(newcell)
            if idx >= len(self.columns):
                newcolumn = TableColumn()

                for row in self.rows:
                    cell = TableCell(width=width, color=color, align=align)
                    row.append(cell)
                    newcolumn.append(cell)
                newcolumn.append(newcell)
                self.columns.append(newcolumn)
            else:
                self.columns[idx].append(newcell)

        for idx in range(len(newrow), len(self.columns)):
            width = widths[idx] if idx < len(widths) else self.columns[idx].width
            color = colors[idx] if idx < len(colors) else ''
            align = aligns[idx] if idx < len(aligns) else ''
            cell = TableCell(width=width, color=color, align=align)
            newrow.append(cell)

        self.rows.append(newrow)
        self._adjust()

    def _adjust(self):
        '''Adjust the width and height of the cells in each row and column.'''
        for column in self.columns:
            _width = -1
            for cell in column:
                _width = max(_width, cell.width)
            for cell in column:
                cell.width = _width

        for row in self.rows:
            _height = -1
            for cell in row:
                _height = max(_height, cell.height)
            for cell in row:
                cell.height = _height

    @property
    def width(self) -> int:
        _width = -1
        for row in self.rows:
            _width = max(_width, row.width)
        return _width

    @property
    def height(self) -> int:
        _height = -1
        for column in self.columns:
            _height = max(_height, column.height)
        return _height

    def __repr__(self) -> str:
        seprow = '+{}+\n'.format('+'.join(['-' * column.width for column in self.columns]))
        content = ''
        for row in self.rows:
            content = content + str(row)
            content += seprow
        return seprow + content


def get_file_logger(filename):
    '''
    Set logger.handler to FileHandler.

    Args:
        filename(str): filename to logging

    Examples:
    .. code-block:: python

        logger = get_file_logger('test.log')
        logger.logger.info('test_1')
    '''
    log_name = os.path.join(LOG_HOME, filename)
    if log_name in loggers:
        return loggers[log_name]

    logger = Logger()
    logger.logger.handlers = []
    format = logging.Formatter('[%(asctime)-15s] [%(levelname)8s] - %(message)s')
    sh = logging.FileHandler(filename=log_name, mode='a')
    sh.setFormatter(format)
    logger.logger.addHandler(sh)
    logger.logger.setLevel(logging.INFO)

    loggers.update({log_name: logger})

    return logger


logger = Logger()
