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

import importlib
import os
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo
from paddlehub.utils.log import logger
from paddlenlp.data import Pad
from parakeet.models import ConditionalWaveFlow, Tacotron2
from parakeet.models.lstm_speaker_encoder import LSTMSpeakerEncoder
import soundfile as sf

from .audio_processor import SpeakerVerificationPreprocessor
from .chinese_g2p import convert_sentence
from .preprocess_transcription import voc_phones, voc_tones, phone_pad_token, tone_pad_token


@moduleinfo(
    name="lstm_tacotron2",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="audio/voice_cloning",
)
class VoiceCloner(nn.Layer):
    def __init__(self, speaker_audio: str = None, output_dir: str = './'):
        super(VoiceCloner, self).__init__()

        self.sample_rate = 22050  # Hyper params for the following model ckpts.
        speaker_encoder_ckpt = os.path.join(MODULE_HOME, 'lstm_tacotron2', 'assets',
                                            'ge2e_ckpt_0.3/step-3000000.pdparams')
        synthesizer_ckpt = os.path.join(MODULE_HOME, 'lstm_tacotron2', 'assets',
                                        'tacotron2_aishell3_ckpt_0.3/step-450000.pdparams')
        vocoder_ckpt = os.path.join(MODULE_HOME, 'lstm_tacotron2', 'assets',
                                    'waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams')

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
        self.synthesizer = Tacotron2(
            vocab_size=68,
            n_tones=10,
            d_mels=80,
            d_encoder=512,
            encoder_conv_layers=3,
            encoder_kernel_size=5,
            d_prenet=256,
            d_attention_rnn=1024,
            d_decoder_rnn=1024,
            attention_filters=32,
            attention_kernel_size=31,
            d_attention=128,
            d_postnet=512,
            postnet_kernel_size=5,
            postnet_conv_layers=5,
            reduction_factor=1,
            p_encoder_dropout=0.5,
            p_prenet_dropout=0.5,
            p_attention_dropout=0.1,
            p_decoder_dropout=0.1,
            p_postnet_dropout=0.5,
            d_global_condition=256,
            use_stop_token=False)
        self.synthesizer.set_state_dict(paddle.load(synthesizer_ckpt))
        self.synthesizer.eval()

        # Vocoder
        self.vocoder = ConditionalWaveFlow(
            upsample_factors=[16, 16], n_flows=8, n_layers=8, n_group=16, channels=128, n_mels=80, kernel_size=[3, 3])
        self.vocoder.set_state_dict(paddle.load(vocoder_ckpt))
        self.vocoder.eval()

        # Speaking embedding
        self._speaker_embedding = None
        if speaker_audio is None or not os.path.isfile(speaker_audio):
            speaker_audio = os.path.join(MODULE_HOME, 'lstm_tacotron2', 'assets', 'voice_cloning.wav')
            logger.warning(f'Due to no speaker audio is specified, speaker encoder will use defult '
                           f'waveform({speaker_audio}) to extract speaker embedding. You can use '
                           '"set_speaker_embedding()" method to reset a speaker audio for voice cloning.')
        self.set_speaker_embedding(speaker_audio)

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_speaker_embedding(self):
        return self._speaker_embedding.numpy()

    def set_speaker_embedding(self, speaker_audio: str):
        assert os.path.exists(speaker_audio), f'Speaker audio file: {speaker_audio} does not exists.'
        mel_sequences = self.speaker_processor.extract_mel_partials(
            self.speaker_processor.preprocess_wav(speaker_audio))
        self._speaker_embedding = self.speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))
        logger.info(f'Speaker embedding has been set from file: {speaker_audio}')

    def forward(self, phones: paddle.Tensor, tones: paddle.Tensor, speaker_embeddings: paddle.Tensor):
        outputs = self.synthesizer.infer(phones, tones=tones, global_condition=speaker_embeddings)
        mel_input = paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1])
        waveforms = self.vocoder.infer(mel_input)
        return waveforms

    def _convert_text_to_input(self, text: str):
        """
        Convert input string to phones and tones.
        """
        phones, tones = convert_sentence(text)
        phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
        tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)
        return phones, tones

    def _batchify(self, data: List[str], batch_size: int):
        """
        Generate input batches.
        """
        phone_pad_func = Pad(voc_phones.lookup(phone_pad_token))
        tone_pad_func = Pad(voc_tones.lookup(tone_pad_token))

        def _parse_batch(batch_data):
            phones, tones = zip(*batch_data)
            speaker_embeddings = paddle.expand(self._speaker_embedding, shape=(len(batch_data), -1))
            return phone_pad_func(phones), tone_pad_func(tones), speaker_embeddings

        examples = []  # [(phones, tones), ...]
        for text in data:
            examples.append(self._convert_text_to_input(text))

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def generate(self, data: List[str], batch_size: int = 1, use_gpu: bool = False):
        assert self._speaker_embedding is not None, f'Set speaker embedding before voice cloning.'

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')
        batches = self._batchify(data, batch_size)

        results = []
        for batch in batches:
            phones, tones, speaker_embeddings = map(paddle.to_tensor, batch)
            waveforms = self(phones, tones, speaker_embeddings).numpy()
            results.extend(list(waveforms))

        files = []
        for idx, waveform in enumerate(results):
            output_wav = os.path.join(self.output_dir, f'{idx+1}.wav')
            sf.write(output_wav, waveform, samplerate=self.sample_rate)
            files.append(output_wav)

        return files
