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
from paddleaudio import load, save_wav
from paddlespeech.cli import ASRExecutor
from paddlehub.module.module import moduleinfo, serving
from paddlehub.utils.log import logger


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
        assert audio_file.endswith('.wav'), 'Input file must be a wave file `*.wav`.'
        sig, sample_rate = load(audio_file)
        if sample_rate != 16000:
            sig, _ = load(audio_file, 16000)
            audio_file_16k = audio_file[:audio_file.rindex('.')] + '_16k.wav'
            logger.info('Resampling to 16000 sample rate to new audio file: {}'.format(audio_file_16k))
            save_wav(sig, 16000, audio_file_16k)
            return audio_file_16k
        else:
            return audio_file

    @serving
    def speech_recognize(self, audio_file, device='cpu'):
        assert os.path.isfile(audio_file), 'File not exists: {}'.format(audio_file)
        audio_file = self.check_audio(audio_file)
        text = self.asr_executor(audio_file=audio_file, device=device, **self.asr_kw_args)
        return text
