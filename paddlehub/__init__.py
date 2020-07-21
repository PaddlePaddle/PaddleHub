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

from .version import hub_version as __version__

from . import module
from . import common
from . import io
from . import dataset
from . import finetune
from . import reader
from . import network
from . import tokenizer

from .common.dir import USER_HOME
from .common.dir import HUB_HOME
from .common.dir import MODULE_HOME
from .common.dir import CACHE_HOME
from .common.dir import CONF_HOME
from .common.logger import logger
from .common.paddle_helper import connect_program
from .common.hub_server import HubServer
from .common.hub_server import server_check
from .common.paddlex_utils import download, ResourceNotFoundError, ServerConnectionError

from .module.module import Module
from .module.base_processor import BaseProcessor
from .module.signature import Signature, create_signature
from .module.manager import default_module_manager

from .io.type import DataType

from .finetune.config import RunConfig
from .finetune.strategy import AdamWeightDecayStrategy, CombinedStrategy, DefaultFinetuneStrategy, DefaultStrategy, L2SPFinetuneStrategy, ULMFiTStrategy
from .finetune.task import BaseTask, ClassifierTask, DetectionTask, TextGenerationTask, ImageClassifierTask, MultiLabelClassifierTask, ReadingComprehensionTask, RegressionTask, SequenceLabelTask, TextClassifierTask, PairwiseTextMatchingTask, PointwiseTextMatchingTask

from .autofinetune.evaluator import report_final_result

from .module.nlp_module import NLPPredictionModule, TransformerModule

from .tokenizer.bert_tokenizer import BertTokenizer, ErnieTinyTokenizer
from .tokenizer.tokenizer import CustomTokenizer
