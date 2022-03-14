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

from .ecapa_tdnn import ECAPA_TDNN
from .feature import compute_log_fbank
from .feature import normalize
from paddlehub.module.module import moduleinfo
from paddlehub.utils.log import logger


@moduleinfo(
    name="ecapa_tdnn_voxceleb",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="audio/speaker_recognition")
class SpeakerRecognition(paddle.nn.Layer):
    def __init__(self, threshold=0.25):
        super(SpeakerRecognition, self).__init__()
        global_stats_path = os.path.join(self.directory, 'assets', 'global_embedding_stats.npy')
        ckpt_path = os.path.join(self.directory, 'assets', 'model.pdparams')

        self.sr = 16000
        self.threshold = threshold
        model_conf = {
            'input_size': 80,
            'channels': [1024, 1024, 1024, 1024, 3072],
            'kernel_sizes': [5, 3, 3, 3, 1],
            'dilations': [1, 2, 3, 4, 1],
            'attention_channels': 128,
            'lin_neurons': 192
        }
        self.model = ECAPA_TDNN(**model_conf)
        self.model.set_state_dict(paddle.load(ckpt_path))
        self.model.eval()

        global_embedding_stats = np.load(global_stats_path, allow_pickle=True)
        self.global_emb_mean = paddle.to_tensor(global_embedding_stats.item().get('global_emb_mean'))
        self.global_emb_std = paddle.to_tensor(global_embedding_stats.item().get('global_emb_std'))

        self.similarity = paddle.nn.CosineSimilarity(axis=-1, eps=1e-6)

    def load_audio(self, wav):
        wav = os.path.abspath(os.path.expanduser(wav))
        assert os.path.isfile(wav), 'Please check wav file: {}'.format(wav)
        waveform, _ = paddleaudio.load(wav, sr=self.sr, mono=True, normal=False)
        return waveform

    def speaker_embedding(self, wav):
        waveform = self.load_audio(wav)
        embedding = self(paddle.to_tensor(waveform)).reshape([-1])
        return embedding.numpy()

    def speaker_verify(self, wav1, wav2):
        waveform1 = self.load_audio(wav1)
        embedding1 = self(paddle.to_tensor(waveform1)).reshape([-1])

        waveform2 = self.load_audio(wav2)
        embedding2 = self(paddle.to_tensor(waveform2)).reshape([-1])

        score = self.similarity(embedding1, embedding2).numpy()
        return score, score > self.threshold

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        fbank = compute_log_fbank(x)  # x: waveform tensors with (B, T) shape
        norm_fbank = normalize(fbank)
        embedding = self.model(norm_fbank.transpose([0, 2, 1])).transpose([0, 2, 1])
        norm_embedding = normalize(x=embedding, global_mean=self.global_emb_mean, global_std=self.global_emb_std)

        return norm_embedding
