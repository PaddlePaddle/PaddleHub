# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
import numpy as np
import matplotlib

matplotlib.use("agg")
import librosa
from scipy import signal
import paddle.fluid.dygraph as dg
from parakeet.g2p import en


def make_evaluator(config, text_sequences):
    c = config["transform"]
    p_replace = 0.0
    sample_rate = c["sample_rate"]
    preemphasis = c["preemphasis"]
    win_length = c["win_length"]
    hop_length = c["hop_length"]
    min_level_db = c["min_level_db"]
    ref_level_db = c["ref_level_db"]

    synthesis_config = config["synthesis"]
    power = synthesis_config["power"]
    n_iter = synthesis_config["n_iter"]

    return Evaluator(text_sequences, p_replace, sample_rate, preemphasis,
                     win_length, hop_length, min_level_db, ref_level_db, power,
                     n_iter)


class Evaluator(object):
    def __init__(self, text_sequences, p_replace, sample_rate, preemphasis,
                 win_length, hop_length, min_level_db, ref_level_db, power,
                 n_iter):
        self.text_sequences = text_sequences

        self.p_replace = p_replace
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.win_length = win_length
        self.hop_length = hop_length
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

        self.power = power
        self.n_iter = n_iter

    def process_a_sentence(self, model, text):
        text = np.array(
            en.text_to_sequence(text, p=self.p_replace), dtype=np.int64)
        length = len(text)
        text_positions = np.arange(1, 1 + length, dtype=np.int64)
        text = np.expand_dims(text, 0)
        text_positions = np.expand_dims(text_positions, 0)

        model.eval()
        if isinstance(model, dg.DataParallel):
            _model = model._layers
        else:
            _model = model
        mel_outputs, linear_outputs, alignments, done = _model.transduce(
            dg.to_variable(text), dg.to_variable(text_positions))

        linear_outputs_np = linear_outputs.numpy()[0].T  # (C, T)

        wav = spec_to_waveform(
            linear_outputs_np, self.min_level_db, self.ref_level_db, self.power,
            self.n_iter, self.win_length, self.hop_length, self.preemphasis)
        alignments_np = alignments.numpy()[0]  # batch_size = 1
        return wav, alignments_np

    def __call__(self, model, iteration):
        wavs = []
        for i, seq in enumerate(self.text_sequences):
            print("[Eval] synthesizing sentence {}".format(i))
            wav, alignments_np = self.process_a_sentence(model, seq)
            wavs.append(wav)
        return wavs, self.sample_rate


def spec_to_waveform(spec, min_level_db, ref_level_db, power, n_iter,
                     win_length, hop_length, preemphasis):
    """Convert output linear spec to waveform using griffin-lim vocoder.

    Args:
        spec (ndarray): the output linear spectrogram, shape(C, T), where C means n_fft, T means frames.
    """
    denoramlized = np.clip(spec, 0, 1) * (-min_level_db) + min_level_db
    lin_scaled = np.exp((denoramlized + ref_level_db) / 20 * np.log(10))
    wav = librosa.griffinlim(
        lin_scaled**power,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length)
    if preemphasis > 0:
        wav = signal.lfilter([1.], [1., -preemphasis], wav)
    wav = np.clip(wav, -1.0, 1.0)
    return wav
