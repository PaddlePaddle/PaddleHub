#coding:utf-8
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
import os
import six

os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"

if six.PY2:
    import sys
    reload(sys)  # noqa
    sys.setdefaultencoding("UTF-8")

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
from .common.dir import CONF_HOME
from .common.logger import logger
from .common.paddle_helper import connect_program
from .common.hub_server import default_hub_server

from .module.module import Module, create_module
from .module.base_processor import BaseProcessor
from .module.signature import Signature, create_signature
from .module.manager import default_module_manager

from .io.type import DataType

from .finetune.task import ClassifierTask
from .finetune.task import TextClassifierTask
from .finetune.task import ImageClassifierTask
from .finetune.task import SequenceLabelTask
from .finetune.task import MultiLabelClassifierTask
from .finetune.task import RegressionTask
from .finetune.task import ReadingComprehensionTask
from .finetune.config import RunConfig
from .finetune.strategy import AdamWeightDecayStrategy
from .finetune.strategy import DefaultStrategy
from .finetune.strategy import DefaultFinetuneStrategy
from .finetune.strategy import L2SPFinetuneStrategy
from .finetune.strategy import ULMFiTStrategy
from .finetune.strategy import CombinedStrategy

from .autofinetune.evaluator import report_final_result

from .common.hub_server import server_check
