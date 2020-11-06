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
import ast
import argparse
import importlib.util

import nltk
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddlehub as hub
from paddlehub.module.module import runnable
from paddlehub.common.utils import mkdir
from paddlehub.module.nlp_module import DataFormatError
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from paddlehub.common.dir import THIRD_PARTY_HOME
from paddlehub.common.downloader import default_downloader

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
    from ruamel import yaml
    from parakeet.models.fastspeech.fastspeech import FastSpeech as FastSpeechModel
    from parakeet.g2p.en import text_to_sequence
    from parakeet.models.transformer_tts.utils import *
    from parakeet.utils import io
    from parakeet.modules.weight_norm import WeightNormWrapper
    from parakeet.models.waveflow import WaveFlowModule
else:
    raise ImportError(
        "The module requires additional dependencies: %s. You can install parakeet via 'git clone https://github.com/PaddlePaddle/Parakeet && cd Parakeet && pip install -e .' and others via pip install"
        % ", ".join(lack_dependency))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@moduleinfo(
    name="fastspeech_ljspeech",
    version="1.0.0",
    summary=
    "FastSpeech proposes a novel feed-forward network based on Transformer to generate mel-spectrogram in parallel for TTS. See https://arxiv.org/abs/1905.09263 for details.",
    author="baidu-nlp",
    author_email="",
    type="nlp/tts",
)
class FastSpeech(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.tts_checkpoint_path = os.path.join(self.directory, "assets", "tts", "step-162000")
        self.waveflow_checkpoint_path = os.path.join(self.directory, "assets", "vocoder", "step-2000000")
        self.waveflow_config_path = os.path.join(self.directory, "assets", "vocoder", "waveflow_ljspeech.yaml")

        tts_config_path = os.path.join(self.directory, "assets", "tts", "ljspeech.yaml")
        with open(tts_config_path) as f:
            self.tts_config = yaml.load(f, Loader=yaml.Loader)
        with fluid.dygraph.guard(fluid.CPUPlace()):
            self.tts_model = FastSpeechModel(self.tts_config['network'], num_mels=self.tts_config['audio']['num_mels'])
            io.load_parameters(model=self.tts_model, checkpoint_path=self.tts_checkpoint_path)

            # Build vocoder.
            args = AttrDict()
            args.config = self.waveflow_config_path
            args.use_fp16 = False
            self.waveflow_config = io.add_yaml_config_to_args(args)
            self.waveflow = WaveFlowModule(self.waveflow_config)
            io.load_parameters(model=self.waveflow, checkpoint_path=self.waveflow_checkpoint_path)

    def synthesize(self, texts, use_gpu=False, speed=1.0, vocoder="griffin-lim"):
        """
        Get the synthetic wavs from the texts.

        Args:
             texts(list): the input texts to be predicted.
             use_gpu(bool): whether use gpu to predict or not. Default False.
             speed(float): Controlling the voice speed. Default 1.0.
             vocoder(str): the vocoder name, "griffin-lim" or "waveflow".

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
            raise ValueError("The input data is inconsistent with expectations.")

        wavs = []
        with fluid.dygraph.guard(place):
            self.tts_model.eval()
            self.waveflow.eval()
            for text in predicted_data:
                # init input
                logger.info("Processing sentence: %s" % text)
                text = np.asarray(text_to_sequence(text))
                text = np.expand_dims(text, axis=0)
                pos_text = np.arange(1, text.shape[1] + 1)
                pos_text = np.expand_dims(pos_text, axis=0)

                text = dg.to_variable(text).astype(np.int64)
                pos_text = dg.to_variable(pos_text).astype(np.int64)

                _, mel_output_postnet = self.tts_model(text, pos_text, alpha=1 / speed)

                if vocoder == 'griffin-lim':
                    # synthesis use griffin-lim
                    wav = self.synthesis_with_griffinlim(mel_output_postnet, self.tts_config['audio'])
                elif vocoder == 'waveflow':
                    wav = self.synthesis_with_waveflow(mel_output_postnet, self.waveflow_config.sigma)
                else:
                    raise ValueError(
                        'vocoder error, we only support griffinlim and waveflow, but recevied %s.' % vocoder)
                wavs.append(wav)
        return wavs, self.tts_config['audio']['sr']

    def synthesis_with_griffinlim(self, mel_output, cfg):
        # synthesis with griffin-lim
        mel_output = fluid.layers.transpose(fluid.layers.squeeze(mel_output, [0]), [1, 0])
        mel_output = np.exp(mel_output.numpy())
        basis = librosa.filters.mel(cfg['sr'], cfg['n_fft'], cfg['num_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'])
        inv_basis = np.linalg.pinv(basis)
        spec = np.maximum(1e-10, np.dot(inv_basis, mel_output))

        wav = librosa.core.griffinlim(spec**cfg['power'], hop_length=cfg['hop_length'], win_length=cfg['win_length'])

        return wav

    def synthesis_with_waveflow(self, mel_output, sigma):
        mel_spectrogram = fluid.layers.transpose(fluid.layers.squeeze(mel_output, [0]), [1, 0])
        mel_spectrogram = fluid.layers.unsqueeze(mel_spectrogram, [0])

        for layer in self.waveflow.sublayers():
            if isinstance(layer, WeightNormWrapper):
                layer.remove_weight_norm()

        # Run model inference.
        wav = self.waveflow.synthesize(mel_spectrogram, sigma=sigma)
        return wav.numpy()[0]

    @serving
    def serving_method(self, texts, use_gpu=False, speed=1.0, vocoder="griffin-lim"):
        """
        Run as a service.
        """
        wavs, sample_rate = self.synthesize(texts, use_gpu, speed, vocoder)
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

    module = FastSpeech()
    test_text = [
        "Simple as this proposition is, it is necessary to be stated",
    ]
    wavs, sample_rate = module.synthesize(texts=test_text, speed=1, vocoder="waveflow")
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
