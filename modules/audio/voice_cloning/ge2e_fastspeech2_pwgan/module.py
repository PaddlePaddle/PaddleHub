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
from typing import List, Union

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo, serving
from paddlehub.utils.log import logger
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Inference
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


@moduleinfo(
    name="ge2e_fastspeech2_pwgan",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="audio/voice_cloning",
)
class VoiceCloner(paddle.nn.Layer):
    def __init__(self, speaker_audio: str = None, output_dir: str = './'):
        super(VoiceCloner, self).__init__()

        speaker_encoder_ckpt = os.path.join(MODULE_HOME, 'ge2e_fastspeech2_pwgan', 'assets',
                                            'ge2e_ckpt_0.3/step-3000000.pdparams')
        synthesizer_res_dir = os.path.join(MODULE_HOME, 'ge2e_fastspeech2_pwgan', 'assets',
                                           'fastspeech2_nosil_aishell3_vc1_ckpt_0.5')
        vocoder_res_dir = os.path.join(MODULE_HOME, 'ge2e_fastspeech2_pwgan', 'assets', 'pwg_aishell3_ckpt_0.5')

        # Speaker encoder
        self.speaker_processor = SpeakerVerificationPreprocessor(
            sampling_rate=16000,
            audio_norm_target_dBFS=-30,
            vad_window_length=30,
            vad_moving_average_width=8,
            vad_max_silence_length=6,
            mel_window_length=25,
            mel_window_step=10,
            n_mels=40,
            partial_n_frames=160,
            min_pad_coverage=0.75,
            partial_overlap_ratio=0.5)
        self.speaker_encoder = LSTMSpeakerEncoder(n_mels=40, num_layers=3, hidden_size=256, output_size=256)
        self.speaker_encoder.set_state_dict(paddle.load(speaker_encoder_ckpt))
        self.speaker_encoder.eval()

        # Voice synthesizer
        with open(os.path.join(synthesizer_res_dir, 'default.yaml'), 'r') as f:
            fastspeech2_config = CfgNode(yaml.safe_load(f))
        with open(os.path.join(synthesizer_res_dir, 'phone_id_map.txt'), 'r') as f:
            phn_id = [line.strip().split() for line in f.readlines()]

        model = FastSpeech2(idim=len(phn_id), odim=fastspeech2_config.n_mels, **fastspeech2_config["model"])
        model.set_state_dict(paddle.load(os.path.join(synthesizer_res_dir, 'snapshot_iter_96400.pdz'))["main_params"])
        model.eval()

        stat = np.load(os.path.join(synthesizer_res_dir, 'speech_stats.npy'))
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        fastspeech2_normalizer = ZScore(mu, std)
        self.sample_rate = fastspeech2_config.fs

        self.fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)
        self.fastspeech2_inference.eval()

        # Vocoder
        with open(os.path.join(vocoder_res_dir, 'default.yaml')) as f:
            pwg_config = CfgNode(yaml.safe_load(f))

        vocoder = PWGGenerator(**pwg_config["generator_params"])
        vocoder.set_state_dict(
            paddle.load(os.path.join(vocoder_res_dir, 'snapshot_iter_1000000.pdz'))["generator_params"])
        vocoder.remove_weight_norm()
        vocoder.eval()

        stat = np.load(os.path.join(vocoder_res_dir, 'feats_stats.npy'))
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        pwg_normalizer = ZScore(mu, std)

        self.pwg_inference = PWGInference(pwg_normalizer, vocoder)
        self.pwg_inference.eval()

        # Text frontend
        self.frontend = Frontend(phone_vocab_path=os.path.join(synthesizer_res_dir, 'phone_id_map.txt'))

        # Speaking embedding
        self._speaker_embedding = None
        if speaker_audio is None or not os.path.isfile(speaker_audio):
            speaker_audio = os.path.join(MODULE_HOME, 'ge2e_fastspeech2_pwgan', 'assets', 'voice_cloning.wav')
            logger.warning(f'Due to no speaker audio is specified, speaker encoder will use defult '
                           f'waveform({speaker_audio}) to extract speaker embedding. You can use '
                           '"set_speaker_embedding()" method to reset a speaker audio for voice cloning.')
        self.set_speaker_embedding(speaker_audio)

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_speaker_embedding(self):
        return self._speaker_embedding.numpy()

    @paddle.no_grad()
    def set_speaker_embedding(self, speaker_audio: str):
        assert os.path.exists(speaker_audio), f'Speaker audio file: {speaker_audio} does not exists.'
        mel_sequences = self.speaker_processor.extract_mel_partials(
            self.speaker_processor.preprocess_wav(speaker_audio))
        self._speaker_embedding = self.speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))

        logger.info(f'Speaker embedding has been set from file: {speaker_audio}')

    @paddle.no_grad()
    def generate(self, data: Union[str, List[str]], use_gpu: bool = False):
        assert self._speaker_embedding is not None, f'Set speaker embedding before voice cloning.'

        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list):
            assert len(data) > 0 and isinstance(data[0],
                                                str) and len(data[0]) > 0, f'Input data should be str of List[str].'
        else:
            raise Exception(f'Input data should be str of List[str].')

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')
        files = []
        for idx, text in enumerate(data):
            phone_ids = self.frontend.get_input_ids(text, merge_sentences=True)["phone_ids"][0]
            wav = self.pwg_inference(self.fastspeech2_inference(phone_ids, spk_emb=self._speaker_embedding))
            output_wav = os.path.join(self.output_dir, f'{idx+1}.wav')
            sf.write(output_wav, wav.numpy(), samplerate=self.sample_rate)
            files.append(output_wav)

        return files
