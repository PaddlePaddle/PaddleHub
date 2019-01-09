#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
from enum import Enum, unique


@unique
class ParamTrainConfig(Enum):
    PARAM_TRAIN_DEFAULT = 0
    PARAM_TRAIN_ALL = 1
    PARAM_TRAIN_NONE = 2


class RunConfig:
    def __init__(self, param_train_config=None):
        assert (not param_train_config or param_train_config in ParamTrainConfig
                ), "train config should be value of %s" % ParamTrainConfig

        if not param_train_config:
            param_train_config = ParamTrainConfig.PARAM_TRAIN_DEFAULT
        self.param_train_config = param_train_config
