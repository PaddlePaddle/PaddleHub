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
from tqdm import tqdm

import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from paddlehub.common.dir import THIRD_PARTY_HOME
from paddlehub.common.downloader import default_downloader

lack_dependency = []
for dependency in ["ruamel", "parakeet", "scipy"]:
    if not importlib.util.find_spec(dependency):
        lack_dependency.append(dependency)

if not lack_dependency:
    from ruamel import yaml
    from scipy.io.wavfile import write
    from parakeet.g2p.en import text_to_sequence
    from parakeet.models.transformer_tts.utils import *
    from parakeet import audio
    from parakeet.models.transformer_tts import TransformerTTS as TransformerTTSModel
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
    name="transformer_tts",
    version="1.0.0",
    summary=
    "Transformer TTS introduces and adapts the multi-head attention mechanism to replace the RNN structures and also the original attention mechanism in Tacotron2. See https://arxiv.org/abs/1809.08895 for details",
    author="baidu-nlp",
    author_email="",
    type="audio/tts",
)
class TransformerTTS(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "step-120000")
        config_path = os.path.join(self.directory, "assets", "config.yaml")
        with open(config_path, "rt") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self._ljspeech_processor = audio.AudioProcessor(
            sample_rate=self.config['audio']['sr'],
            num_mels=self.config['audio']['num_mels'],
            min_level_db=self.config['audio']['min_level_db'],
            ref_level_db=self.config['audio']['ref_level_db'],
            n_fft=self.config['audio']['n_fft'],
            win_length=self.config['audio']['win_length'],
            hop_length=self.config['audio']['hop_length'],
            power=self.config['audio']['power'],
            preemphasis=self.config['audio']['preemphasis'],
            signal_norm=True,
            symmetric_norm=False,
            max_norm=1.,
            mel_fmin=0,
            mel_fmax=None,
            clip_norm=True,
            griffin_lim_iters=60,
            do_trim_silence=False,
            sound_norm=False)

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
        with fluid.unique_name.guard():
            network_cfg = self.config['network']
            model = TransformerTTSModel(network_cfg['embedding_size'],
                                        network_cfg['hidden_size'],
                                        network_cfg['encoder_num_head'],
                                        network_cfg['encoder_n_layers'],
                                        self.config['audio']['num_mels'],
                                        network_cfg['outputs_per_step'],
                                        network_cfg['decoder_num_head'],
                                        network_cfg['decoder_n_layers'])
            # Load parameters.
            global_step = io.load_parameters(
                model=model, checkpoint_path=self.pretrained_model_path)
            model.eval()

        wavs = []
        for text in predicted_data:
            # init input
            audio_len = len(text.split()) * 40  # An empirical value
            text = np.asarray(text_to_sequence(text))
            text = fluid.layers.unsqueeze(
                dg.to_variable(text).astype(np.int64), [0])
            mel_input = dg.to_variable(np.zeros([1, 1, 80])).astype(np.float32)
            pos_text = np.arange(1, text.shape[1] + 1)
            pos_text = fluid.layers.unsqueeze(
                dg.to_variable(pos_text).astype(np.int64), [0])

            for _ in tqdm(range(audio_len)):
                pos_mel = np.arange(1, mel_input.shape[1] + 1)
                pos_mel = fluid.layers.unsqueeze(
                    dg.to_variable(pos_mel).astype(np.int64), [0])
                mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
                    text, mel_input, pos_text, pos_mel)
                mel_input = fluid.layers.concat(
                    [mel_input, postnet_pred[:, -1:, :]], axis=1)

            # synthesis with griffin-lim
            wav = self._ljspeech_processor.inv_melspectrogram(
                fluid.layers.transpose(
                    fluid.layers.squeeze(postnet_pred, [0]), [1, 0]).numpy())
            wavs.append(wav)
        return wavs, self.config['audio']['sr']


if __name__ == "__main__":
    import soundfile as sf

    module = TransformerTTS()
    test_text = [
        "hello, how are you", "Hello, how do you do",
        "Parakeet stands for Paddle PARAllel text-to-speech toolkit."
    ]
    wavs, sample_rate = module.synthesize(texts=test_text)
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
