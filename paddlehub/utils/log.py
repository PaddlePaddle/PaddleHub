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

import colorlog

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


class TableLogger(object):
    def __init__(self, head=None, head_colors=None, head_aligns=None, colors=None, aligns=None, slots_len=None):
        self.contents = []
        self.slot_nums = -1
        self.head = head
        self.head_colors = head_colors
        self.head_aligns = head_aligns
        self.colors = colors
        self.aligns = aligns
        self.slots_len = slots_len

    def __enter__(self):
        return self

    def __exit__(self, exit_exception, exit_value, exit_traceback):
        if not exit_value:
            self._print()

    def add(self, *line):
        self.contents.append(line)
        self.slot_nums = max(self.slot_nums, len(line))

    def _print(self):
        slots_len = [-1] * self.slot_nums

        for line in self.contents:
            for idx, item in enumerate(line):
                slots_len[idx] = max(slots_len[idx], len(item))

        formats = ['{' + ':<{}'.format(_len) + '}' for _len in slots_len]

        print('+'.join(['-' * _len for _len in slots_len]))

        for line in self.contents:
            print('|'.join([formats[idx].format(item) for idx, item in enumerate(line)]))
            print('+'.join(['-' * _len for _len in slots_len]))


logger = Logger()
