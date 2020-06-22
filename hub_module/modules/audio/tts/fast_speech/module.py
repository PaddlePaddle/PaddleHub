# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib.util

import nltk
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from paddlehub.common.dir import THIRD_PARTY_HOME
from paddlehub.common.downloader import default_downloader

lack_dependency = []
for dependency in ["ruamel", "parakeet"]:
    if not importlib.util.find_spec(dependency):
        lack_dependency.append(dependency)

if not lack_dependency:
    from ruamel import yaml
    from parakeet.models.fastspeech.fastspeech import FastSpeech as FastSpeechModel
    from parakeet.g2p.en import text_to_sequence
    from parakeet.models.transformer_tts.utils import *
    from parakeet.utils import io
else:
    raise ImportError(
        "The module requires additional dependencies. Please install %s via `pip install`"
        % ", ".join(lack_dependency))

# Accelerate NLTK package download via paddlehub
_PUNKT_URL = "https://paddlehub.bj.bcebos.com/paddlehub-thirdparty/punkt.tar.gz"
_CMUDICT_URL = "https://paddlehub.bj.bcebos.com/paddlehub-thirdparty/cmudict.tar.gz"
nltk_path = os.path.join(THIRD_PARTY_HOME, "nltk_data")
tokenizers_path = os.path.join(nltk_path, "tokenizers")
corpora_path = os.path.join(nltk_path, "corpora")
punkt_path = os.path.join(tokenizers_path, "punkt")
cmudict_path = os.path.join(corpora_path, "cmudict")

if not os.path.exists(punkt_path):
    default_downloader.download_file_and_uncompress(
        url=_PUNKT_URL, save_path=tokenizers_path, print_progress=True)
# if not os.path.exists(cmudict_path):
#     default_downloader.download_file_and_uncompress(
#         url=_CMUDICT_URL, save_path=corpora_path, print_progress=True)
nltk.data.path.append(nltk_path)


@moduleinfo(
    name="fast_speech",
    version="1.0.0",
    summary=
    "FastSpeech proposes a novel feed-forward network based on Transformer to generate mel-spectrogram in parallel for TTS. See https://arxiv.org/abs/1905.09263 for details.",
    author="baidu-nlp",
    author_email="",
    type="audio/tts",
)
class FastSpeech(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "step-130000")
        config_path = os.path.join(self.directory, "assets", "config.yaml")
        with open(config_path, "rt") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    @serving
    def synthesize(self, texts, use_gpu=False, accelerate_rate=1.0):
        """
        Get the sentiment prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted, if texts not data
             use_gpu(bool): whether use gpu to predict or not. Default False.
             accelerate_rate(float): Controlling the voice speed. Default 1.0.

        Returns:
             wavs(str): the audio wav with sample rate . You can use soundfile.write to save it.
             sample_rate(int): the audio sample rate.
        """
        if use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
            use_gpu = False
            logger.warning(
                "use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True"
            )
        if use_gpu:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        if texts and isinstance(texts, list):
            predicted_data = texts
        else:
            raise ValueError(
                "The input data is inconsistent with expectations.")
        dg.enable_dygraph(place)

        model = FastSpeechModel(
            self.config['network'], num_mels=self.config['audio']['num_mels'])

        # Load parameters.
        global_step = io.load_parameters(
            model=model, checkpoint_path=self.pretrained_model_path)
        model.eval()

        wavs = []
        for text in predicted_data:
            # init input
            logger.info("Processing sentence: %s" % text)
            text = np.asarray(text_to_sequence(text))
            text = np.expand_dims(text, axis=0)
            pos_text = np.arange(1, text.shape[1] + 1)
            pos_text = np.expand_dims(pos_text, axis=0)

            text = dg.to_variable(text).astype(np.int64)
            pos_text = dg.to_variable(pos_text).astype(np.int64)

            _, mel_output_postnet = model(
                text, pos_text, alpha=1 / accelerate_rate)

            mel_output_postnet = fluid.layers.transpose(
                fluid.layers.squeeze(mel_output_postnet, [0]), [1, 0])
            mel_output_postnet = np.exp(mel_output_postnet.numpy())
            basis = librosa.filters.mel(self.config['audio']['sr'],
                                        self.config['audio']['n_fft'],
                                        self.config['audio']['num_mels'])
            inv_basis = np.linalg.pinv(basis)
            spec = np.maximum(1e-10, np.dot(inv_basis, mel_output_postnet))

            # synthesis with griffin-lim
            wav = librosa.core.griffinlim(
                spec**self.config['audio']['power'],
                hop_length=self.config['audio']['hop_length'],
                win_length=self.config['audio']['win_length'])
            wavs.append(wav)
        return wavs, self.config['audio']['sr']


if __name__ == "__main__":
    import soundfile as sf

    module = FastSpeech()
    test_text = [
        "Simple as this proposition is, it is necessary to be stated,",
        "hello, how are you",
        "Hello, how do you do",
        "Parakeet stands for Paddle PARAllel text-to-speech toolkit.",
    ]
    wavs, sample_rate = module.synthesize(texts=test_text, accelerate_rate=2)
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
