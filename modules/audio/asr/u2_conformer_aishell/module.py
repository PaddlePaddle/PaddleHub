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
import sys

import numpy as np
from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo, serving
from paddlehub.utils.log import logger

import paddle
import soundfile as sf

# TODO: Remove system path when deepspeech can be installed via pip.
sys.path.append(os.path.join(MODULE_HOME, 'u2_conformer_aishell'))
from deepspeech.exps.u2.config import get_cfg_defaults
from deepspeech.utils.utility import UpdateConfig
from .u2_conformer_tester import U2ConformerTester


@moduleinfo(name="u2_conformer_aishell", version="1.0.0", summary="", author="Baidu", author_email="", type="audio/asr")
class U2Conformer(paddle.nn.Layer):
    def __init__(self):
        super(U2Conformer, self).__init__()

        # resource
        res_dir = os.path.join(MODULE_HOME, 'u2_conformer_aishell', 'assets')
        conf_file = os.path.join(res_dir, 'conf/conformer.yaml')
        checkpoint = os.path.join(res_dir, 'checkpoints/avg_20.pdparams')

        # config
        self.config = get_cfg_defaults()
        self.config.merge_from_file(conf_file)

        # TODO: Remove path updating snippet.
        with UpdateConfig(self.config):
            self.config.collator.vocab_filepath = os.path.join(res_dir, self.config.collator.vocab_filepath)
            # self.config.collator.spm_model_prefix = os.path.join(res_dir, self.config.collator.spm_model_prefix)
            self.config.collator.augmentation_config = os.path.join(res_dir, self.config.collator.augmentation_config)
            self.config.model.cmvn_file = os.path.join(res_dir, self.config.model.cmvn_file)
            self.config.decoding.decoding_method = 'attention_rescoring'
            self.config.decoding.batch_size = 1

        # model
        self.tester = U2ConformerTester(self.config)
        self.tester.setup_model()
        self.tester.resume(checkpoint)

    @staticmethod
    def check_audio(audio_file):
        sig, sample_rate = sf.read(audio_file)
        assert sample_rate == 16000, 'Excepting sample rate of input audio is 16000, but got {}'.format(sample_rate)

    @serving
    def speech_recognize(self, audio_file, device='cpu'):
        assert os.path.isfile(audio_file), 'File not exists: {}'.format(audio_file)
        self.check_audio(audio_file)

        paddle.set_device(device)
        return self.tester.test(audio_file)[0][0]
