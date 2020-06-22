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

import os
import importlib.util

import nltk
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from paddlehub.common.dir import THIRD_PARTY_HOME
from paddlehub.common.downloader import default_downloader

from deep_voice3.model import make_model
from deep_voice3.utils import make_evaluator

lack_dependency = []
for dependency in ["ruamel", "parakeet"]:
    if not importlib.util.find_spec(dependency):
        lack_dependency.append(dependency)

if not lack_dependency:
    import ruamel.yaml
    from parakeet.modules.weight_norm import WeightNormWrapper
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
    name="deep_voice3",
    version="1.0.0",
    summary=
    "Deep Voice 3, a fully-convolutional attention-based neural text-to-speech (TTS) system.",
    author="baidu-nlp",
    author_email="",
    type="audio/tts",
)
class DeepVoice3(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "step-1560000")
        config_path = os.path.join(self.directory, "assets", "config.yaml")
        with open(config_path, "rt") as f:
            self.config = ruamel.yaml.safe_load(f)

    @serving
    def synthesize(self, texts, use_gpu=False):
        """
        Get the sentiment prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted, if texts not data
             use_gpu(bool): whether use gpu to predict or not

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
        model = make_model(self.config)
        iteration = io.load_parameters(
            model, checkpoint_path=self.pretrained_model_path)

        for layer in model.sublayers():
            if isinstance(layer, WeightNormWrapper):
                layer.remove_weight_norm()

        evaluator = make_evaluator(self.config, predicted_data)
        wavs = evaluator(model, iteration)
        return wavs


if __name__ == "__main__":
    import soundfile as sf

    module = DeepVoice3()
    test_text = ["hello, how are you", "Hello, how do you do"]
    wavs, sample_rate = module.synthesize(texts=test_text)
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
