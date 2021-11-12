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
from pathlib import Path
from typing import List

import numpy as np
import paddle
from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo, serving
from paddlehub.utils.log import logger
from parakeet.frontend import English
from parakeet.models.fastspeech2 import FastSpeech2
from parakeet.models.fastspeech2 import FastSpeech2Inference
from parakeet.models.parallel_wavegan import PWGGenerator
from parakeet.models.parallel_wavegan import PWGInference
from parakeet.modules.normalizer import ZScore
import soundfile as sf
from yacs.config import CfgNode
import yaml


@moduleinfo(name="fastspeech2_ljspeech", version="1.0.0", summary="", author="Baidu", author_email="", type="audio/tts")
class FastSpeech(paddle.nn.Layer):
    def __init__(self, output_dir='./wavs'):
        super(FastSpeech, self).__init__()
        fastspeech2_res_dir = os.path.join(MODULE_HOME, 'fastspeech2_ljspeech',
                                           'assets/fastspeech2_nosil_ljspeech_ckpt_0.5')
        pwg_res_dir = os.path.join(MODULE_HOME, 'fastspeech2_ljspeech', 'assets/pwg_ljspeech_ckpt_0.5')

        phones_dict = os.path.join(fastspeech2_res_dir, 'phone_id_map.txt')
        with open(phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        self.phone_id_map = {}
        for phn, _id in phn_id:
            self.phone_id_map[phn] = int(_id)

        # fastspeech2
        fastspeech2_config = os.path.join(fastspeech2_res_dir, 'default.yaml')
        with open(fastspeech2_config) as f:
            fastspeech2_config = CfgNode(yaml.safe_load(f))
        self.samplerate = fastspeech2_config.fs

        fastspeech2_checkpoint = os.path.join(fastspeech2_res_dir, 'snapshot_iter_100000.pdz')
        model = FastSpeech2(idim=vocab_size, odim=fastspeech2_config.n_mels, **fastspeech2_config["model"])
        model.set_state_dict(paddle.load(fastspeech2_checkpoint)["main_params"])
        logger.info('Load fastspeech2 params from %s' % os.path.abspath(fastspeech2_checkpoint))
        model.eval()

        # vocoder
        pwg_config = os.path.join(pwg_res_dir, 'pwg_default.yaml')
        with open(pwg_config) as f:
            pwg_config = CfgNode(yaml.safe_load(f))

        pwg_checkpoint = os.path.join(pwg_res_dir, 'pwg_snapshot_iter_400000.pdz')
        vocoder = PWGGenerator(**pwg_config["generator_params"])
        vocoder.set_state_dict(paddle.load(pwg_checkpoint)["generator_params"])
        logger.info('Load vocoder params from %s' % os.path.abspath(pwg_checkpoint))
        vocoder.remove_weight_norm()
        vocoder.eval()

        # frontend
        self.frontend = English()
        self.punc = "：，；。？！“”‘’':,;.?!"

        # stat
        fastspeech2_stat = os.path.join(fastspeech2_res_dir, 'speech_stats.npy')
        stat = np.load(fastspeech2_stat)
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        fastspeech2_normalizer = ZScore(mu, std)

        pwg_stat = os.path.join(pwg_res_dir, 'pwg_stats.npy')
        stat = np.load(pwg_stat)
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        pwg_normalizer = ZScore(mu, std)

        # inference
        self.fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)
        self.pwg_inference = PWGInference(pwg_normalizer, vocoder)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, text: str):
        phones = self.frontend.phoneticize(text)
        # remove start_symbol and end_symbol
        phones = phones[1:-1]
        phones = [phn for phn in phones if not phn.isspace()]
        phones = [phn if (phn in self.phone_id_map and phn not in self.punc) else "sp" for phn in phones]
        phone_ids = [self.phone_id_map[phn] for phn in phones]
        phone_ids = paddle.to_tensor(phone_ids)

        with paddle.no_grad():
            mel = self.fastspeech2_inference(phone_ids)
            wav = self.pwg_inference(mel)

        return wav

    @serving
    def generate(self, sentences: List[str], device='cpu'):
        assert isinstance(sentences, list) and isinstance(sentences[0], str), \
            'Input data should be List[str], but got {}'.format(type(sentences))

        paddle.set_device(device)
        wav_files = []
        for i, sentence in enumerate(sentences):
            wav = self(sentence)
            wav_file = str(self.output_dir.absolute() / (str(i + 1) + ".wav"))
            sf.write(wav_file, wav.numpy(), samplerate=self.samplerate)
            wav_files.append(wav_file)

        logger.info('{} wave files have been generated in {}'.format(len(sentences), self.output_dir.absolute()))
        return wav_files
