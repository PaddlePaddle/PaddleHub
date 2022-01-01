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

import numpy as np
import paddle
import paddleaudio

from .feature import compute_mfcc
from .kwmlp import KW_MLP
from paddlehub.module.module import moduleinfo
from paddlehub.utils.log import logger


@moduleinfo(
    name="kwmlp_speech_commands",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="audio/language_identification")
class KWS(paddle.nn.Layer):
    def __init__(self):
        super(KWS, self).__init__()
        ckpt_path = os.path.join(self.directory, 'assets', 'model.pdparams')
        label_path = os.path.join(self.directory, 'assets', 'label.txt')

        self.label_list = []
        with open(label_path, 'r') as f:
            for l in f:
                self.label_list.append(l.strip())

        self.sr = 16000
        model_conf = {
            'input_res': [40, 98],
            'patch_res': [40, 1],
            'num_classes': 35,
            'channels': 1,
            'dim': 64,
            'depth': 12,
            'pre_norm': False,
            'prob_survival': 0.9,
        }
        self.model = KW_MLP(**model_conf)
        self.model.set_state_dict(paddle.load(ckpt_path))
        self.model.eval()

    def load_audio(self, wav):
        wav = os.path.abspath(os.path.expanduser(wav))
        assert os.path.isfile(wav), 'Please check wav file: {}'.format(wav)
        waveform, _ = paddleaudio.load(wav, sr=self.sr, mono=True, normal=False)
        return waveform

    def keyword_recognize(self, wav):
        waveform = self.load_audio(wav)

        # fix_length to 1s
        if len(waveform) > self.sr:
            waveform = waveform[:self.sr]
        else:
            waveform = np.pad(waveform, (0, self.sr - len(waveform)))

        logits = self(paddle.to_tensor(waveform)).reshape([-1])
        probs = paddle.nn.functional.softmax(logits)
        idx = paddle.argmax(probs)
        return probs[idx].numpy(), self.label_list[idx]

    def forward(self, x):
        if len(x.shape) == 1:  # x: waveform tensors with (B, T) shape
            x = x.unsqueeze(0)

        mfcc = compute_mfcc(x).unsqueeze(1)  # (B, C, n_mels, L)
        logits = self.model(mfcc).squeeze(1)

        return logits
