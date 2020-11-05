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

import sys

from easydict import EasyDict

__version__ = '2.0.0-alpha0'

from paddlehub.config import config
from paddlehub.utils import log, parser, utils
from paddlehub.utils.paddlex import download, ResourceNotFoundError
from paddlehub.server.server_source import ServerConnectionError
from paddlehub.module import Module
from paddlehub.text.bert_tokenizer import BertTokenizer

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
sys.modules['paddlehub.common.logger'] = log
sys.modules['paddlehub.common.paddle_helper'] = paddle_utils
sys.modules['paddlehub.common.utils'] = utils
sys.modules['paddlehub.reader'] = task

common = EasyDict(paddle_helper=paddle_utils)
dataset = EasyDict(Couplet=couplet.Couplet)
AdamWeightDecayStrategy = lambda: 0
