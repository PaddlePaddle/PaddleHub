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
"""Evaluation for DeepSpeech2 model."""
import os
import sys
from pathlib import Path

import paddle

from deepspeech.frontend.featurizer.text_featurizer import TextFeaturizer
from deepspeech.io.collator import SpeechCollator
from deepspeech.models.ds2 import DeepSpeech2Model
from deepspeech.utils import mp_tools
from deepspeech.utils.utility import UpdateConfig


class DeepSpeech2Tester:
    def __init__(self, config):
        self.config = config
        self.collate_fn_test = SpeechCollator.from_config(config)
        self._text_featurizer = TextFeaturizer(unit_type=config.collator.unit_type, vocab_filepath=None)

    def compute_result_transcripts(self, audio, audio_len, vocab_list, cfg):
        result_transcripts = self.model.decode(
            audio,
            audio_len,
            vocab_list,
            decoding_method=cfg.decoding_method,
            lang_model_path=cfg.lang_model_path,
            beam_alpha=cfg.alpha,
            beam_beta=cfg.beta,
            beam_size=cfg.beam_size,
            cutoff_prob=cfg.cutoff_prob,
            cutoff_top_n=cfg.cutoff_top_n,
            num_processes=cfg.num_proc_bsearch)
        #replace the '<space>' with ' '
        result_transcripts = [self._text_featurizer.detokenize(sentence) for sentence in result_transcripts]

        return result_transcripts

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self, audio_file):
        self.model.eval()
        cfg = self.config
        collate_fn_test = self.collate_fn_test
        audio, _ = collate_fn_test.process_utterance(audio_file=audio_file, transcript=" ")
        audio_len = audio.shape[0]
        audio = paddle.to_tensor(audio, dtype='float32')
        audio_len = paddle.to_tensor(audio_len)
        audio = paddle.unsqueeze(audio, axis=0)
        vocab_list = collate_fn_test.vocab_list
        result_transcripts = self.compute_result_transcripts(audio, audio_len, vocab_list, cfg.decoding)
        return result_transcripts

    def setup_model(self):
        config = self.config.clone()
        with UpdateConfig(config):
            config.model.feat_size = self.collate_fn_test.feature_size
            config.model.dict_size = self.collate_fn_test.vocab_size

        model = DeepSpeech2Model.from_config(config.model)
        self.model = model

    def resume(self, checkpoint):
        """Resume from the checkpoint at checkpoints in the output
        directory or load a specified checkpoint.
        """
        model_dict = paddle.load(checkpoint)
        self.model.set_state_dict(model_dict)
