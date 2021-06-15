# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import math
import os
from typing import Dict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from panns_cnn10.network import CNN10

from paddlehub.env import MODULE_HOME
from paddlehub.module.audio_module import AudioClassifierModule
from paddlehub.module.module import moduleinfo
from paddlehub.utils.log import logger


@moduleinfo(
    name="panns_cnn10",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="audio/sound_classification",
    meta=AudioClassifierModule)
class PANN(nn.Layer):
    def __init__(
            self,
            task: str,
            num_class: int = None,
            label_map: Dict = None,
            load_checkpoint: str = None,
            **kwargs,
    ):
        super(PANN, self).__init__()

        if label_map:
            self.label_map = label_map
            self.num_class = len(label_map)
        else:
            self.num_class = num_class

        if task == 'sound-cls':
            self.cnn10 = CNN10(
                extract_embedding=True, checkpoint=os.path.join(MODULE_HOME, 'panns_cnn10', 'cnn10.pdparams'))
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(self.cnn10.emb_size, num_class)
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy()
        else:
            self.cnn10 = CNN10(
                extract_embedding=False, checkpoint=os.path.join(MODULE_HOME, 'panns_cnn10', 'cnn10.pdparams'))

        self.task = task
        if load_checkpoint is not None and os.path.isfile(load_checkpoint):
            state_dict = paddle.load(load_checkpoint)
            self.set_state_dict(state_dict)
            logger.info('Loaded parameters from %s' % os.path.abspath(load_checkpoint))

    def forward(self, feats, labels=None):
        # feats: (batch_size, num_frames, num_melbins) -> (batch_size, 1, num_frames, num_melbins)
        feats = feats.unsqueeze(1)

        if self.task == 'sound-cls':
            embeddings = self.cnn10(feats)
            embeddings = self.dropout(embeddings)
            logits = self.fc(embeddings)
            probs = F.softmax(logits, axis=1)

            if labels is not None:
                loss = self.criterion(logits, labels)
                correct = self.metric.compute(probs, labels)
                acc = self.metric.update(correct)
                return probs, loss, {'acc': acc}
            return probs
        else:
            audioset_logits = self.cnn10(feats)
            return audioset_logits
