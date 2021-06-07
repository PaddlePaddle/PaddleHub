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
import numpy as np
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

lack_dependency = []
for dependency in ["ruamel", "parakeet", "soundfile", "librosa"]:
    if not importlib.util.find_spec(dependency):
        lack_dependency.append(dependency)

# Accelerate NLTK package download via paddlehub. 'import parakeet' will use the package.
_PUNKT_URL = "https://paddlehub.bj.bcebos.com/paddlehub-thirdparty/punkt.tar.gz"
_CMUDICT_URL = "https://paddlehub.bj.bcebos.com/paddlehub-thirdparty/cmudict.tar.gz"
nltk_path = os.path.join(THIRD_PARTY_HOME, "nltk_data")
tokenizers_path = os.path.join(nltk_path, "tokenizers")
corpora_path = os.path.join(nltk_path, "corpora")
punkt_path = os.path.join(tokenizers_path, "punkt")
cmudict_path = os.path.join(corpora_path, "cmudict")

if not os.path.exists(punkt_path):
    default_downloader.download_file_and_uncompress(url=_PUNKT_URL, save_path=tokenizers_path, print_progress=True)
if not os.path.exists(cmudict_path):
    default_downloader.download_file_and_uncompress(url=_CMUDICT_URL, save_path=corpora_path, print_progress=True)
nltk.data.path.append(nltk_path)

if not lack_dependency:
    import soundfile as sf
    import librosa
    import ruamel.yaml
    from parakeet.utils import io
    from parakeet.g2p import en
    from parakeet.models.deepvoice3 import Encoder, Decoder, PostNet, SpectraNet
    from parakeet.models.waveflow import WaveFlowModule
    from parakeet.models.deepvoice3.weight_norm_hook import remove_weight_norm
else:
    raise ImportError(
        "The module requires additional dependencies: %s. You can install parakeet via 'git clone https://github.com/PaddlePaddle/Parakeet && cd Parakeet && pip install -e .' and others via pip install"
        % ", ".join(lack_dependency))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class WaveflowVocoder(object):
    def __init__(self, config_path, checkpoint_path):
        with open(config_path, 'rt') as f:
            config = ruamel.yaml.safe_load(f)
        ns = argparse.Namespace()
        for k, v in config.items():
            setattr(ns, k, v)
        ns.use_fp16 = False

        self.model = WaveFlowModule(ns)
        io.load_parameters(self.model, checkpoint_path=checkpoint_path)

    def __call__(self, mel):
        with dg.no_grad():
            self.model.eval()
            audio = self.model.synthesize(mel)
        self.model.train()
        return audio


class GriffinLimVocoder(object):
    def __init__(self, sharpening_factor=1.4, sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.sharpening_factor = sharpening_factor
        self.win_length = win_length
        self.hop_length = hop_length

    def __call__(self, mel):
        spec = librosa.feature.inverse.mel_to_stft(
            np.exp(mel), sr=self.sample_rate, n_fft=self.n_fft, fmin=0, fmax=8000.0, power=1.0)
        audio = librosa.core.griffinlim(
            spec**self.sharpening_factor, win_length=self.win_length, hop_length=self.hop_length)
        return audio


@moduleinfo(
    name="deepvoice3_ljspeech",
    version="1.0.0",
    summary="Deep Voice 3, a fully-convolutional attention-based neural text-to-speech (TTS) system.",
    author="paddlepaddle",
    author_email="",
    type="nlp/tts",
)
class DeepVoice3(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.tts_checkpoint_path = os.path.join(self.directory, "assets", "tts", "step-1780000")
        self.waveflow_checkpoint_path = os.path.join(self.directory, "assets", "vocoder", "step-2000000")
        self.waveflow_config_path = os.path.join(self.directory, "assets", "vocoder", "waveflow_ljspeech.yaml")
        tts_checkpoint_path = os.path.join(self.directory, "assets", "tts", "ljspeech.yaml")
        with open(tts_checkpoint_path) as f:
            self.tts_config = ruamel.yaml.safe_load(f)

        with fluid.dygraph.guard(fluid.CPUPlace()):
            char_embedding = dg.Embedding((en.n_vocab, self.tts_config["char_dim"]))
            multi_speaker = self.tts_config["n_speakers"] > 1
            speaker_embedding = dg.Embedding((self.tts_config["n_speakers"], self.tts_config["speaker_dim"])) \
                if multi_speaker else None
            encoder = Encoder(
                self.tts_config["encoder_layers"],
                self.tts_config["char_dim"],
                self.tts_config["encoder_dim"],
                self.tts_config["kernel_size"],
                has_bias=multi_speaker,
                bias_dim=self.tts_config["speaker_dim"],
                keep_prob=1.0 - self.tts_config["dropout"])
            decoder = Decoder(
                self.tts_config["n_mels"],
                self.tts_config["reduction_factor"],
                list(self.tts_config["prenet_sizes"]) + [self.tts_config["char_dim"]],
                self.tts_config["decoder_layers"],
                self.tts_config["kernel_size"],
                self.tts_config["attention_dim"],
                position_encoding_weight=self.tts_config["position_weight"],
                omega=self.tts_config["position_rate"],
                has_bias=multi_speaker,
                bias_dim=self.tts_config["speaker_dim"],
                keep_prob=1.0 - self.tts_config["dropout"])
            postnet = PostNet(
                self.tts_config["postnet_layers"],
                self.tts_config["char_dim"],
                self.tts_config["postnet_dim"],
                self.tts_config["kernel_size"],
                self.tts_config["n_mels"],
                self.tts_config["reduction_factor"],
                has_bias=multi_speaker,
                bias_dim=self.tts_config["speaker_dim"],
                keep_prob=1.0 - self.tts_config["dropout"])
            self.tts_model = SpectraNet(char_embedding, speaker_embedding, encoder, decoder, postnet)
            io.load_parameters(model=self.tts_model, checkpoint_path=self.tts_checkpoint_path)
            for name, layer in self.tts_model.named_sublayers():
                try:
                    remove_weight_norm(layer)
                except ValueError:
                    # this layer has not weight norm hook
                    pass

            self.waveflow = WaveflowVocoder(
                config_path=self.waveflow_config_path, checkpoint_path=self.waveflow_checkpoint_path)
            self.griffin = GriffinLimVocoder(
                sharpening_factor=self.tts_config["sharpening_factor"],
                sample_rate=self.tts_config["sample_rate"],
                n_fft=self.tts_config["n_fft"],
                win_length=self.tts_config["win_length"],
                hop_length=self.tts_config["hop_length"])

    def synthesize(self, texts, use_gpu=False, vocoder="griffin-lim"):
        """
        Get the synthetic wavs from the texts.

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
            raise ValueError("The input data is inconsistent with expectations.")

        wavs = []
        with fluid.dygraph.guard(place):
            self.tts_model.eval()
            self.waveflow.model.eval()
            monotonic_layers = [4]
            for text in predicted_data:
                # init input
                logger.info("Processing sentence: %s" % text)
                text = en.text_to_sequence(text, p=1.0)
                text = np.expand_dims(np.array(text, dtype="int64"), 0)
                lengths = np.array([text.size], dtype=np.int64)
                text_seqs = dg.to_variable(text)
                text_lengths = dg.to_variable(lengths)

                decoder_layers = self.tts_config["decoder_layers"]
                force_monotonic_attention = [False] * decoder_layers
                for i in monotonic_layers:
                    force_monotonic_attention[i] = True

                outputs = self.tts_model(
                    text_seqs,
                    text_lengths,
                    speakers=None,
                    force_monotonic_attention=force_monotonic_attention,
                    window=(self.tts_config["backward_step"], self.tts_config["forward_step"]))
                decoded, refined, attentions = outputs
                if vocoder == 'griffin-lim':
                    # synthesis use griffin-lim
                    wav = self.griffin(refined.numpy()[0].T)
                elif vocoder == 'waveflow':
                    # synthesis use waveflow
                    wav = self.waveflow(fluid.layers.transpose(refined, [0, 2, 1])).numpy()[0]
                else:
                    raise ValueError(
                        'vocoder error, we only support griffinlim and waveflow, but recevied %s.' % vocoder)
                wavs.append(wav)
        return wavs, self.tts_config["sample_rate"]

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
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU for prediction")

        self.arg_config_group.add_argument(
            '--vocoder', type=str, default="griffin-lim", choices=['griffin-lim', 'waveflow'], help="the vocoder name")

    def add_module_output_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--output_path',
            type=str,
            default=os.path.abspath(os.path.join(os.path.curdir, f"{self.name}_prediction")),
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

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_input_group = self.parser.add_argument_group(
            title="Ouput options", description="Ouput path. Optional.")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, optional.")

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
        wavs, sample_rate = self.synthesize(texts=input_data, use_gpu=args.use_gpu, vocoder=args.vocoder)

        for index, wav in enumerate(wavs):
            sf.write(os.path.join(args.output_path, f"{index}.wav"), wav, sample_rate)

        ret = f"The synthesized wav files have been saved in {args.output_path}"
        return ret


if __name__ == "__main__":
    module = DeepVoice3()
    test_text = [
        "Simple as this proposition is, it is necessary to be stated",
        "Parakeet stands for Paddle PARAllel text-to-speech toolkit.",
    ]
    wavs, sample_rate = module.synthesize(texts=test_text, vocoder="waveflow")
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
