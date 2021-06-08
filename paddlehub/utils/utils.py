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

import base64
import contextlib
import hashlib
import importlib
import math
import os
import socket
import sys
import tempfile
import time
import traceback
import types
from typing import Generator, List
from urllib.parse import urlparse

import cv2
import numpy as np
import packaging.version
import requests

import paddlehub.env as hubenv
import paddlehub.utils as utils
from paddlehub.utils.log import logger


class Version(packaging.version.Version):
    '''Extended implementation of packaging.version.Version'''

    def match(self, condition: str) -> bool:
        '''
        Determine whether the given condition are met
        Args:
            condition(str) : conditions for judgment
        Returns:
            bool: True if the given version condition are met, else False
        Examples:
            .. code-block:: python
                Version('1.2.0').match('>=1.2.0a')
        '''
        if not condition:
            return True
        if condition.startswith('>='):
            version = condition[2:]
            _comp = self.__ge__
        elif condition.startswith('>'):
            version = condition[1:]
            _comp = self.__gt__
        elif condition.startswith('<='):
            version = condition[2:]
            _comp = self.__le__
        elif condition.startswith('<'):
            version = condition[1:]
            _comp = self.__lt__
        elif condition.startswith('=='):
            version = condition[2:]
            _comp = self.__eq__
        elif condition.startswith('='):
            version = condition[1:]
            _comp = self.__eq__
        else:
            version = condition
            _comp = self.__eq__

        return _comp(Version(version))

    def __lt__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__ge__(other)

    def __eq__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__eq__(other)


class Timer(object):
    '''Calculate runing speed and estimated time of arrival(ETA)'''

    def __init__(self, total_step: int):
        self.total_step = total_step
        self.last_start_step = 0
        self.current_step = 0
        self._is_running = True

    def start(self):
        self.last_time = time.time()
        self.start_time = time.time()

    def stop(self):
        self._is_running = False
        self.end_time = time.time()

    def count(self) -> int:
        if not self.current_step >= self.total_step:
            self.current_step += 1
        return self.current_step

    @property
    def timing(self) -> float:
        run_steps = self.current_step - self.last_start_step
        self.last_start_step = self.current_step
        time_used = time.time() - self.last_time
        self.last_time = time.time()
        return run_steps / time_used

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def eta(self) -> str:
        if not self.is_running:
            return '00:00:00'
        scale = self.total_step / self.current_step
        remaining_time = (time.time() - self.start_time) * scale
        return seconds_to_hms(remaining_time)


def seconds_to_hms(seconds: int) -> str:
    '''Convert the number of seconds to hh:mm:ss'''
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = '{:0>2}:{:0>2}:{:0>2}'.format(h, m, s)
    return hms_str


def cv2_to_base64(image: np.ndarray) -> str:
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str: str) -> np.ndarray:
    '''Convert a string in base64 format to cv2 data'''
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


@contextlib.contextmanager
def generate_tempfile(directory: str = None, **kwargs):
    '''Generate a temporary file'''
    directory = hubenv.TMP_HOME if not directory else directory
    with tempfile.NamedTemporaryFile(dir=directory, **kwargs) as file:
        yield file


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    '''Generate a temporary directory'''
    directory = hubenv.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def download(url: str, path: str = None) -> str:
    '''
    Download a file
    Args:
        url (str) : url to be downloaded
        path (str, optional) : path to store downloaded products, default is current work directory
    Examples:
        .. code-block:: python
            url = 'https://xxxxx.xx/xx.tar.gz'
            download(url, path='./output')
    '''
    for savename, _, _ in download_with_progress(url, path):
        ...
    return savename


def download_with_progress(url: str, path: str = None) -> Generator[str, int, int]:
    '''
    Download a file and return the downloading progress -> Generator[filename, download_size, total_size]
    Args:
        url (str) : url to be downloaded
        path (str, optional) : path to store downloaded products, default is current work directory
    Examples:
        .. code-block:: python
            url = 'https://xxxxx.xx/xx.tar.gz'
            for filename, download_size, total_szie in download_with_progress(url, path='./output'):
                print(filename, download_size, total_size)
    '''
    path = os.getcwd() if not path else path
    if not os.path.exists(path):
        os.makedirs(path)

    parse_result = urlparse(url)
    savename = parse_result.path.split('/')[-1]
    savename = os.path.join(path, savename)

    res = requests.get(url, stream=True)
    download_size = 0
    total_size = int(res.headers.get('content-length'))
    with open(savename, 'wb') as _file:
        for data in res.iter_content(chunk_size=4096):
            _file.write(data)
            download_size += len(data)
            yield savename, download_size, total_size


def load_py_module(python_path: str, py_module_name: str) -> types.ModuleType:
    '''
    Load the specified python module.

    Args:
        python_path(str) : The directory where the python module is located
        py_module_name(str) : Module name to be loaded
    '''
    sys.path.insert(0, python_path)

    # Delete the cache module to avoid hazards. For example, when the user reinstalls a HubModule,
    # if the cache is not cleared, then what the user gets at this time is actually the HubModule
    # before uninstallation, this can cause some strange problems, e.g, fail to load model parameters.
    if py_module_name in sys.modules:
        sys.modules.pop(py_module_name)

    py_module = importlib.import_module(py_module_name)
    sys.path.pop(0)

    return py_module


def get_platform_default_encoding() -> str:
    '''Get the default encoding of the current platform.'''
    if utils.platform.is_windows():
        return 'gbk'
    return 'utf8'


def sys_stdin_encoding() -> str:
    '''Get the standary input stream default encoding.'''
    encoding = sys.stdin.encoding
    if encoding is None:
        encoding = sys.getdefaultencoding()

    if encoding is None:
        encoding = get_platform_default_encoding()
    return encoding


def sys_stdout_encoding() -> str:
    '''Get the standary output stream default encoding.'''
    encoding = sys.stdout.encoding
    if encoding is None:
        encoding = sys.getdefaultencoding()

    if encoding is None:
        encoding = get_platform_default_encoding()
    return encoding


def md5(text: str):
    '''Calculate the md5 value of the input text.'''
    md5code = hashlib.md5(text.encode())
    return md5code.hexdigest()


def record(msg: str) -> str:
    '''Record the specified text into the PaddleHub log file witch will be automatically stored according to date.'''
    logfile = get_record_file()
    with open(logfile, 'a') as file:
        file.write('=' * 50 + '\n')
        file.write('Record at ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        file.write('=' * 50 + '\n')
        file.write(str(msg) + '\n' * 3)

    return logfile


def record_exception(msg: str) -> str:
    '''Record the current exception infomation into the PaddleHub log file witch will be automatically stored according to date.'''
    tb = traceback.format_exc()
    file = record(tb)
    logger.warning('{}. Detailed error information can be found in the {}.'.format(msg, file))


def get_record_file() -> str:
    return os.path.join(hubenv.LOG_HOME, time.strftime('%Y%m%d.log'))


def is_port_occupied(ip: str, port: int) -> bool:
    '''
    Check if port os occupied.
    '''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def mkdir(path: str):
    """The same as the shell command `mkdir -p`."""
    if not os.path.exists(path):
        os.makedirs(path)


def reseg_token_label(tokenizer, tokens: List[str], labels: List[str] = None):
    '''
    Convert segments and labels of sequence labeling samples into tokens
    based on the vocab of tokenizer.
    '''
    if labels:
        if len(tokens) != len(labels):
            raise ValueError("The length of tokens must be same with labels")
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer._tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            ret_labels.append(label)
            if len(sub_token) < 2:
                continue
            sub_label = label
            if label.startswith("B-"):
                sub_label = "I-" + label[2:]
            ret_labels.extend([sub_label] * (len(sub_token) - 1))

        if len(ret_tokens) != len(ret_labels):
            raise ValueError("The length of ret_tokens can't match with labels")
        return ret_tokens, ret_labels
    else:
        ret_tokens = []
        for token in tokens:
            sub_token = tokenizer._tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) < 2:
                continue
        return ret_tokens, None


def pad_sequence(ids: List[int], max_seq_len: int, pad_token_id: int):
    '''
    Pads a sequence to max_seq_len
    '''
    assert len(ids) <= max_seq_len, \
        f'The input length {len(ids)} is greater than max_seq_len {max_seq_len}. '\
        'Please check the input list and max_seq_len if you really want to pad a sequence.'
    return ids[:] + [pad_token_id] * (max_seq_len - len(ids))


def trunc_sequence(ids: List[int], max_seq_len: int):
    '''
    Truncates a sequence to max_seq_len
    '''
    assert len(ids) >= max_seq_len, \
        f'The input length {len(ids)} is less than max_seq_len {max_seq_len}. ' \
        'Please check the input list and max_seq_len if you really want to truncate a sequence.'
    return ids[:max_seq_len]


def extract_melspectrogram(y,
                           sample_rate: int = 32000,
                           window_size: int = 1024,
                           hop_size: int = 320,
                           mel_bins: int = 64,
                           fmin: int = 50,
                           fmax: int = 14000,
                           window: str = 'hann',
                           center: bool = True,
                           pad_mode: str = 'reflect',
                           ref: float = 1.0,
                           amin: float = 1e-10,
                           top_db: float = None):
    '''
    Extract Mel Spectrogram from a waveform.
    '''
    try:
        import librosa
    except Exception:
        logger.error('Failed to import librosa. Please check that librosa and numba are correctly installed.')
        raise

    s = librosa.stft(
        y,
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode)

    power = np.abs(s)**2
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax)
    mel = np.matmul(melW, power)
    db = librosa.power_to_db(mel, ref=ref, amin=amin, top_db=None)
    db = db.transpose()
    return db
