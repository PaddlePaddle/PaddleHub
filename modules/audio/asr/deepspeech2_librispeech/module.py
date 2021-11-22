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
from paddle.utils.download import get_path_from_url

try:
    import swig_decoders
except ModuleNotFoundError as e:
    logger.error(e)
    logger.info('The module requires additional dependencies: swig_decoders. '
                'please install via:\n\'git clone https://github.com/PaddlePaddle/DeepSpeech.git '
                '&& cd DeepSpeech && git reset --hard b53171694e7b87abe7ea96870b2f4d8e0e2b1485 '
                '&& cd deepspeech/decoders/ctcdecoder/swig && sh setup.sh\'')
    sys.exit(1)

import paddle
import soundfile as sf

# TODO: Remove system path when deepspeech can be installed via pip.
sys.path.append(os.path.join(MODULE_HOME, 'deepspeech2_librispeech'))
from deepspeech.exps.deepspeech2.config import get_cfg_defaults
from deepspeech.utils.utility import UpdateConfig
from .deepspeech_tester import DeepSpeech2Tester

LM_URL = 'https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm'
LM_MD5 = '099a601759d467cd0a8523ff939819c5'


@moduleinfo(
    name="deepspeech2_librispeech", version="1.0.0", summary="", author="Baidu", author_email="", type="audio/asr")
class DeepSpeech2(paddle.nn.Layer):
    def __init__(self):
        super(DeepSpeech2, self).__init__()

        # resource
        res_dir = os.path.join(MODULE_HOME, 'deepspeech2_librispeech', 'assets')
        conf_file = os.path.join(res_dir, 'conf/deepspeech2.yaml')
        checkpoint = os.path.join(res_dir, 'checkpoints/avg_1.pdparams')
        # Download LM manually cause its large size.
        lm_path = os.path.join(res_dir, 'data', 'lm')
        lm_file = os.path.join(lm_path, LM_URL.split('/')[-1])
        if not os.path.isfile(lm_file):
            logger.info(f'Downloading lm from {LM_URL}.')
            get_path_from_url(url=LM_URL, root_dir=lm_path, md5sum=LM_MD5)

        # config
        self.model_type = 'offline'
        self.config = get_cfg_defaults(self.model_type)
        self.config.merge_from_file(conf_file)

        # TODO: Remove path updating snippet.
        with UpdateConfig(self.config):
            self.config.collator.mean_std_filepath = os.path.join(res_dir, self.config.collator.mean_std_filepath)
            self.config.collator.vocab_filepath = os.path.join(res_dir, self.config.collator.vocab_filepath)
            self.config.collator.augmentation_config = os.path.join(res_dir, self.config.collator.augmentation_config)
            self.config.decoding.lang_model_path = os.path.join(res_dir, self.config.decoding.lang_model_path)

        # model
        self.tester = DeepSpeech2Tester(self.config)
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
        return self.tester.test(audio_file)[0]
