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

__version__ = 'develop'

import paddle
from packaging.version import Version
_paddle_version = Version(paddle.__version__)
if _paddle_version < Version('2.0.0') and _paddle_version != Version('0.0.0'):
    raise RuntimeError(
        'Version mismatch in PaddleHub and PaddlePaddle, you need to upgrade PaddlePaddle to version 2.0.0 or above.')

import sys

from easydict import EasyDict

from paddlehub import env
from paddlehub.config import config
from paddlehub import datasets
from paddlehub.finetune import Trainer
from paddlehub.utils import log, parser, utils
from paddlehub.utils import download as _download
from paddlehub.utils.paddlex import download, ResourceNotFoundError
from paddlehub.utils.platform import is_windows
from paddlehub.server import server_check
from paddlehub.server.server_source import ServerConnectionError
from paddlehub.module import Module
from paddlehub.text.bert_tokenizer import BertTokenizer, ErnieTinyTokenizer
from paddlehub.text.tokenizer import CustomTokenizer
from paddlehub.text.utils import is_chinese_char

# In order to maintain the compatibility of the old version, we put the relevant
# compatible code in the paddlehub.compat package, and mapped some modules referenced
# in the old version
from paddlehub.compat import paddle_utils
from paddlehub.compat.module.processor import BaseProcessor
from paddlehub.compat.module.nlp_module import NLPPredictionModule, TransformerModule
from paddlehub.compat.type import DataType
from paddlehub.compat import task
from paddlehub.compat.datasets import couplet
from paddlehub.compat.task.config import RunConfig
from paddlehub.compat.task.text_generation_task import TextGenerationTask

sys.modules['paddlehub.io.parser'] = parser
sys.modules['paddlehub.common.dir'] = env
sys.modules['paddlehub.common.downloader'] = _download
sys.modules['paddlehub.common.logger'] = log
sys.modules['paddlehub.common.paddle_helper'] = paddle_utils
sys.modules['paddlehub.common.utils'] = utils
sys.modules['paddlehub.reader'] = task
sys.modules['paddlehub.reader.batching'] = task.batch

AdamWeightDecayStrategy = lambda: 0
ULMFiTStrategy = lambda params_layer=0: 0
common = EasyDict(paddle_helper=paddle_utils)
dataset = EasyDict(Couplet=couplet.Couplet)
finetune = EasyDict(strategy=EasyDict(ULMFiTStrategy=ULMFiTStrategy))
logger = EasyDict(logger=log.logger)


# Alias for paddle.hub.*
def load(*args, **kwargs):
    if _paddle_version < Version('2.1.0') and _paddle_version != Version('0.0.0'):
        raise RuntimeError(
            '`hub.load` is only available in PaddlePaddle 2.1 and above, please upgrade the PaddlePaddle version.')

    from paddle.hub import load as phload
    from paddlehub.server.server import CacheUpdater

    CacheUpdater("paddle.hub.load").start()
    return phload(*args, **kwargs)


def list(*args, **kwargs):
    if _paddle_version < Version('2.1.0') and _paddle_version != Version('0.0.0'):
        raise RuntimeError(
            '`hub.list` is only available in PaddlePaddle 2.1 and above, please upgrade the PaddlePaddle version.')

    from paddle.hub import list as phlist
    from paddlehub.server.server import CacheUpdater

    CacheUpdater("paddle.hub.list").start()
    return phlist(*args, **kwargs)


def help(*args, **kwargs):
    if _paddle_version < Version('2.1.0') and _paddle_version != Version('0.0.0'):
        raise RuntimeError(
            '`hub.help` is only available in PaddlePaddle 2.1 and above, please upgrade the PaddlePaddle version.')

    from paddle.hub import help as phhelp
    from paddlehub.server.server import CacheUpdater

    CacheUpdater("paddle.hub.help").start()
    return phhelp(*args, **kwargs)


if is_windows():
    for char in env.HUB_HOME:
        if is_chinese_char(char):
            log.logger.warning(
                'The home directory contains Chinese characters which may cause unknown exceptions in the execution \
                    of some modules. Please set another path through the set HUB_HOME command.')
            break
