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

#coding:utf-8

import six

from . import module
from . import common
from . import io
from . import dataset
from . import finetune
from . import reader

from .common.dir import USER_HOME
from .common.dir import HUB_HOME
from .common.dir import MODULE_HOME
from .common.dir import CACHE_HOME
from .common.logger import logger
from .common.paddle_helper import connect_program
from .common.hub_server import default_hub_server

from .module.module import Module, create_module
from .module.base_processor import BaseProcessor
from .module.signature import Signature, create_signature
from .module.manager import default_module_manager

from .io.type import DataType

from .finetune.task import Task
from .finetune.task import create_seq_label_task
from .finetune.task import create_text_cls_task
from .finetune.task import create_img_cls_task
from .finetune.finetune import finetune_and_eval
from .finetune.config import RunConfig
from .finetune.strategy import AdamWeightDecayStrategy
from .finetune.strategy import DefaultStrategy

if six.PY2:
    import sys
    reload(sys)
    sys.setdefaultencoding("UTF-8")
