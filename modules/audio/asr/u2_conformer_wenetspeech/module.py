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

import paddle
from paddleaudio import load
from paddlespeech.cli import ASRExecutor
from paddlehub.module.module import moduleinfo, serving


@moduleinfo(
    name="u2_conformer_wenetspeech", version="1.0.0", summary="", author="Wenet", author_email="", type="audio/asr")
class U2Conformer(paddle.nn.Layer):
    def __init__(self):
        super(U2Conformer, self).__init__()
        self.asr_executor = ASRExecutor()
        self.asr_kw_args = {
            'model': 'conformer_wenetspeech',
            'lang': 'zh',
            'sample_rate': 16000,
            'config': None,  # Set `config` and `ckpt_path` to None to use pretrained model.
            'ckpt_path': None,
        }

    @staticmethod
    def check_audio(audio_file):
        sig, sample_rate = load(audio_file)
        assert sample_rate == 16000, 'Excepting sample rate of input audio is 16000, but got {}'.format(sample_rate)

    @serving
    def speech_recognize(self, audio_file, device='cpu'):
        assert os.path.isfile(audio_file), 'File not exists: {}'.format(audio_file)
        self.check_audio(audio_file)
        text = self.asr_executor(audio_file=audio_file, device=device, **self.asr_kw_args)
        return text
