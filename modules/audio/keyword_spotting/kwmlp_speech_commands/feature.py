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

import numpy as np
import paddle
import paddleaudio


def create_dct(n_mfcc: int, n_mels: int, norm: str = 'ortho'):
    n = paddle.arange(float(n_mels))
    k = paddle.arange(float(n_mfcc)).unsqueeze(1)
    dct = paddle.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == "ortho"
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def compute_mfcc(
        x: paddle.Tensor,
        sr: int = 16000,
        n_mels: int = 40,
        n_fft: int = 480,
        win_length: int = 480,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: float = None,
        center: bool = False,
        top_db: float = 80.0,
        norm: str = 'ortho',
):
    fbank = paddleaudio.features.spectrum.MelSpectrogram(
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=0.0,
        f_max=f_max,
        center=center)(x)  # waveforms batch ~ (B, T)
    log_fbank = paddleaudio.features.spectrum.power_to_db(fbank, top_db=top_db)
    dct_matrix = create_dct(n_mfcc=n_mels, n_mels=n_mels, norm=norm)
    mfcc = paddle.matmul(log_fbank.transpose((0, 2, 1)), dct_matrix).transpose((0, 2, 1))  # (B, n_mels, L)
    return mfcc
