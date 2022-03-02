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
"""Evaluation for U2 model."""
import os
import sys

import paddle

from deepspeech.frontend.featurizer.text_featurizer import TextFeaturizer
from deepspeech.io.collator import SpeechCollator
from deepspeech.models.u2 import U2Model
from deepspeech.utils import mp_tools
from deepspeech.utils.utility import UpdateConfig


class U2ConformerTester:
    def __init__(self, config):
        self.config = config
        self.collate_fn_test = SpeechCollator.from_config(config)
        self._text_featurizer = TextFeaturizer(
            unit_type=config.collator.unit_type, vocab_filepath=None, spm_model_prefix=config.collator.spm_model_prefix)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self, audio_file):
        self.model.eval()
        cfg = self.config.decoding
        collate_fn_test = self.collate_fn_test
        audio, _ = collate_fn_test.process_utterance(audio_file=audio_file, transcript="Hello")
        audio_len = audio.shape[0]
        audio = paddle.to_tensor(audio, dtype='float32')
        audio_len = paddle.to_tensor(audio_len)
        audio = paddle.unsqueeze(audio, axis=0)
        vocab_list = collate_fn_test.vocab_list

        text_feature = self.collate_fn_test.text_feature
        result_transcripts = self.model.decode(
            audio,
            audio_len,
            text_feature=text_feature,
            decoding_method=cfg.decoding_method,
            lang_model_path=cfg.lang_model_path,
            beam_alpha=cfg.alpha,
            beam_beta=cfg.beta,
            beam_size=cfg.beam_size,
            cutoff_prob=cfg.cutoff_prob,
            cutoff_top_n=cfg.cutoff_top_n,
            num_processes=cfg.num_proc_bsearch,
            ctc_weight=cfg.ctc_weight,
            decoding_chunk_size=cfg.decoding_chunk_size,
            num_decoding_left_chunks=cfg.num_decoding_left_chunks,
            simulate_streaming=cfg.simulate_streaming)

        return result_transcripts

    def setup_model(self):
        config = self.config.clone()
        with UpdateConfig(config):
            config.model.input_dim = self.collate_fn_test.feature_size
            config.model.output_dim = self.collate_fn_test.vocab_size

        self.model = U2Model.from_config(config.model)

    def resume(self, checkpoint):
        """Resume from the checkpoint at checkpoints in the output
        directory or load a specified checkpoint.
        """
        model_dict = paddle.load(checkpoint)
        self.model.set_state_dict(model_dict)
