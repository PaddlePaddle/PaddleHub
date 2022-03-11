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
import os
import re
from typing import List
from typing import Union

import numpy as np
import paddle
import paddleaudio

from .ecapa_tdnn import Classifier
from .ecapa_tdnn import ECAPA_TDNN
from .feature import compute_log_fbank
from .feature import normalize
from paddlehub.module.module import moduleinfo
from paddlehub.utils.log import logger


@moduleinfo(
    name="ecapa_tdnn_common_language",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="audio/language_identification")
class LanguageIdentification(paddle.nn.Layer):
    def __init__(self):
        super(LanguageIdentification, self).__init__()
        ckpt_path = os.path.join(self.directory, 'assets', 'model.pdparams')
        label_path = os.path.join(self.directory, 'assets', 'label.txt')

        self.label_list = []
        with open(label_path, 'r') as f:
            for l in f:
                self.label_list.append(l.strip())

        self.sr = 16000
        model_conf = {
            'input_size': 80,
            'channels': [1024, 1024, 1024, 1024, 3072],
            'kernel_sizes': [5, 3, 3, 3, 1],
            'dilations': [1, 2, 3, 4, 1],
            'attention_channels': 128,
            'lin_neurons': 192
        }
        self.model = Classifier(
            backbone=ECAPA_TDNN(**model_conf),
            num_class=45,
        )
        self.model.set_state_dict(paddle.load(ckpt_path))
        self.model.eval()

    def load_audio(self, wav):
        wav = os.path.abspath(os.path.expanduser(wav))
        assert os.path.isfile(wav), 'Please check wav file: {}'.format(wav)
        waveform, _ = paddleaudio.load(wav, sr=self.sr, mono=True, normal=False)
        return waveform

    def language_identify(self, wav):
        waveform = self.load_audio(wav)
        logits = self(paddle.to_tensor(waveform)).reshape([-1])
        idx = paddle.argmax(logits)
        return logits[idx].numpy(), self.label_list[idx]

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        fbank = compute_log_fbank(x)  # x: waveform tensors with (B, T) shape
        norm_fbank = normalize(fbank)
        logits = self.model(norm_fbank).squeeze(1)

        return logits
