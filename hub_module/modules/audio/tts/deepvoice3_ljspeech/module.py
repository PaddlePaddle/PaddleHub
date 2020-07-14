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
import os
import argparse
import ast
import importlib.util

import nltk
import soundfile as sf
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from paddlehub.common.dir import THIRD_PARTY_HOME
from paddlehub.common.utils import mkdir
from paddlehub.common.downloader import default_downloader
from paddlehub.module.module import runnable
from paddlehub.module.nlp_module import DataFormatError

from deepvoice3_ljspeech.model import make_model
from deepvoice3_ljspeech.utils import make_evaluator

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
if not os.path.exists(cmudict_path):
    default_downloader.download_file_and_uncompress(
        url=_CMUDICT_URL, save_path=corpora_path, print_progress=True)
nltk.data.path.append(nltk_path)


@moduleinfo(
    name="deepvoice3_ljspeech",
    version="1.0.0",
    summary=
    "Deep Voice 3, a fully-convolutional attention-based neural text-to-speech (TTS) system.",
    author="paddlepaddle",
    author_email="",
    type="nlp/tts",
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

    def synthesize(self, texts, use_gpu=False, vocoder="griffin-lim"):
        """
        Get the sentiment prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted.
             use_gpu(bool): whether use gpu to predict or not
             vocoder(str): the vocoder name, "griffin-lim" or "waveflow"

        Returns:
             wavs(str): the audio wav with sample rate . You can use soundfile.write to save it.
             sample_rate(int): the audio sample rate.
        """
        if use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
            use_gpu = False
            logger.warning(
                "use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True"
            )

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

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
        wavs, sample_rate = evaluator(model, iteration)
        return wavs, sample_rate

    @serving
    def serving_method(self, texts, use_gpu=False, vocoder="griffin-lim"):
        """
        Run as a service.
        """
        wavs, sample_rate = self.synthesize(texts, use_gpu, vocoder)
        wavs = [wav.tolist() for wav in wavs]
        result = {"wavs": wavs, "sample_rate": sample_rate}
        return result

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU for prediction")

        self.arg_config_group.add_argument(
            '--vocoder',
            type=str,
            default="griffin-lim",
            choices=['griffin-lim', 'waveflow'],
            help="the vocoder name")

    def add_module_output_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--output_path',
            type=str,
            default=os.path.abspath(
                os.path.join(os.path.curdir, f"{self.name}_prediction")),
            help="path to save experiment results")

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_input_group = self.parser.add_argument_group(
            title="Ouput options", description="Ouput path. Optional.")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, optional.")

        self.add_module_config_arg()
        self.add_module_input_arg()
        self.add_module_output_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        mkdir(args.output_path)
        wavs, sample_rate = self.synthesize(
            texts=input_data, use_gpu=args.use_gpu, vocoder=args.vocoder)

        for index, wav in enumerate(wavs):
            sf.write(
                os.path.join(args.output_path, f"{index}.wav"), wav,
                sample_rate)

        ret = f"The synthesized wav files have been saved in {args.output_path}"
        return ret


if __name__ == "__main__":
    module = DeepVoice3()
    test_text = [
        "Simple as this proposition is, it is necessary to be stated",
        "Parakeet stands for Paddle PARAllel text-to-speech toolkit.",
    ]
    wavs, sample_rate = module.synthesize(
        texts=test_text, vocoder="griffin-lim")
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
