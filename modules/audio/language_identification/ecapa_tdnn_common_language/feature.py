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
import paddle
import paddleaudio
from paddleaudio.features.spectrum import hz_to_mel
from paddleaudio.features.spectrum import mel_to_hz
from paddleaudio.features.spectrum import power_to_db
from paddleaudio.features.spectrum import Spectrogram
from paddleaudio.features.window import get_window


def compute_fbank_matrix(sample_rate: int = 16000,
                         n_fft: int = 400,
                         n_mels: int = 80,
                         f_min: int = 0.0,
                         f_max: int = 8000.0):
    mel = paddle.linspace(hz_to_mel(f_min, htk=True), hz_to_mel(f_max, htk=True), n_mels + 2, dtype=paddle.float32)
    hz = mel_to_hz(mel, htk=True)

    band = hz[1:] - hz[:-1]
    band = band[:-1]
    f_central = hz[1:-1]

    n_stft = n_fft // 2 + 1
    all_freqs = paddle.linspace(0, sample_rate // 2, n_stft)
    all_freqs_mat = all_freqs.tile([f_central.shape[0], 1])

    f_central_mat = f_central.tile([all_freqs_mat.shape[1], 1]).transpose([1, 0])
    band_mat = band.tile([all_freqs_mat.shape[1], 1]).transpose([1, 0])

    slope = (all_freqs_mat - f_central_mat) / band_mat
    left_side = slope + 1.0
    right_side = -slope + 1.0

    fbank_matrix = paddle.maximum(paddle.zeros_like(left_side), paddle.minimum(left_side, right_side))

    return fbank_matrix


def compute_log_fbank(
        x: paddle.Tensor,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        window: str = 'hamming',
        center: bool = True,
        pad_mode: str = 'constant',
        f_min: float = 0.0,
        f_max: float = None,
        top_db: float = 80.0,
):

    if f_max is None:
        f_max = sample_rate / 2

    spect = Spectrogram(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)(x)

    fbank_matrix = compute_fbank_matrix(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    fbank = paddle.matmul(fbank_matrix, spect)
    log_fbank = power_to_db(fbank, top_db=top_db).transpose([0, 2, 1])
    return log_fbank


def compute_stats(x: paddle.Tensor, mean_norm: bool = True, std_norm: bool = False, eps: float = 1e-10):
    if mean_norm:
        current_mean = paddle.mean(x, axis=0)
    else:
        current_mean = paddle.to_tensor([0.0])

    if std_norm:
        current_std = paddle.std(x, axis=0)
    else:
        current_std = paddle.to_tensor([1.0])

    current_std = paddle.maximum(current_std, eps * paddle.ones_like(current_std))

    return current_mean, current_std


def normalize(
        x: paddle.Tensor,
        global_mean: paddle.Tensor = None,
        global_std: paddle.Tensor = None,
):

    for i in range(x.shape[0]):  # (B, ...)
        if global_mean is None and global_std is None:
            mean, std = compute_stats(x[i])
            x[i] = (x[i] - mean) / std
        else:
            x[i] = (x[i] - global_mean) / global_std
    return x
