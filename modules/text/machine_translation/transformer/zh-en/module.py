# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from typing import List

import paddle
import paddle.nn as nn
from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo, serving
import paddlenlp
from paddlenlp.data import Pad, Vocab
from paddlenlp.transformers import InferTransformerModel, position_encoding_init

from transformer_zh_en.utils import MTTokenizer, post_process_seq


@moduleinfo(
    name="transformer_zh-en",
    version="1.0.1",
    summary="",
    author="PaddlePaddle",
    author_email="",
    type="nlp/machine_translation",
)
class MTTransformer(nn.Layer):
    """
    Transformer model for machine translation.
    """
    # Language config
    lang_config = {'source': 'zh', 'target': 'en'}

    # Model config
    model_config = {
        # Number of head used in multi-head attention.
        "n_head": 8,
        # The dimension for word embeddings, which is also the last dimension of
        # the input and output of multi-head attention, position-wise feed-forward
        # networks, encoder and decoder.
        "d_model": 512,
        # Size of the hidden layer in position-wise feed-forward networks.
        "d_inner_hid": 2048,
        # The flag indicating whether to share embedding and softmax weights.
        # Vocabularies in source and target should be same for weight sharing.
        "weight_sharing": False,
        # Dropout rate
        'dropout': 0,
        # Number of sub-layers to be stacked in the encoder and decoder.
        "num_encoder_layers": 6, 
        "num_decoder_layers": 6
    }

    # Vocab config
    vocab_config = {
        # Used to pad vocab size to be multiple of pad_factor.
        "pad_factor": 8,
        # Index for <bos> token
        "bos_id": 0,
        "bos_token": "<s>",
        # Index for <eos> token
        "eos_id": 1,
        "eos_token": "<e>",
        # Index for <unk> token
        "unk_id": 2,
        "unk_token": "<unk>",
    }

    def __init__(self, max_length: int = 256, max_out_len: int = 256, beam_size: int = 5):
        super(MTTransformer, self).__init__()
        bpe_codes_file = os.path.join(MODULE_HOME, 'transformer_zh_en', 'assets', '2M.zh2en.dict4bpe.zh')
        src_vocab_file = os.path.join(MODULE_HOME, 'transformer_zh_en', 'assets', 'vocab.zh')
        trg_vocab_file = os.path.join(MODULE_HOME, 'transformer_zh_en', 'assets', 'vocab.en')
        checkpoint = os.path.join(MODULE_HOME, 'transformer_zh_en', 'assets', 'transformer.pdparams')

        self.max_length = max_length
        self.beam_size = beam_size
        self.tokenizer = MTTokenizer(
            bpe_codes_file=bpe_codes_file, lang_src=self.lang_config['source'], lang_trg=self.lang_config['target'])
        self.src_vocab = Vocab.load_vocabulary(
            filepath=src_vocab_file,
            unk_token=self.vocab_config['unk_token'],
            bos_token=self.vocab_config['bos_token'],
            eos_token=self.vocab_config['eos_token'])
        self.trg_vocab = Vocab.load_vocabulary(
            filepath=trg_vocab_file,
            unk_token=self.vocab_config['unk_token'],
            bos_token=self.vocab_config['bos_token'],
            eos_token=self.vocab_config['eos_token'])
        self.src_vocab_size = (len(self.src_vocab) + self.vocab_config['pad_factor'] - 1) \
            // self.vocab_config['pad_factor'] * self.vocab_config['pad_factor']
        self.trg_vocab_size = (len(self.trg_vocab) + self.vocab_config['pad_factor'] - 1) \
            // self.vocab_config['pad_factor'] * self.vocab_config['pad_factor']
        self.transformer = InferTransformerModel(
            src_vocab_size=self.src_vocab_size,
            trg_vocab_size=self.trg_vocab_size,
            bos_id=self.vocab_config['bos_id'],
            eos_id=self.vocab_config['eos_id'],
            max_length=self.max_length + 1,
            max_out_len=max_out_len,
            beam_size=self.beam_size,
            **self.model_config)

        state_dict = paddle.load(checkpoint)

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        state_dict["encoder.pos_encoder.weight"] = position_encoding_init(self.max_length + 1,
                                                                          self.model_config['d_model'])
        state_dict["decoder.pos_encoder.weight"] = position_encoding_init(self.max_length + 1,
                                                                          self.model_config['d_model'])

        self.transformer.set_state_dict(state_dict)

    def forward(self, src_words: paddle.Tensor):
        return self.transformer(src_words)

    def _convert_text_to_input(self, text: str):
        """
        Convert input string to ids.
        """
        bpe_tokens = self.tokenizer.tokenize(text)
        if len(bpe_tokens) > self.max_length:
            bpe_tokens = bpe_tokens[:self.max_length]
        return self.src_vocab.to_indices(bpe_tokens)

    def _batchify(self, data: List[str], batch_size: int):
        """
        Generate input batches.
        """
        pad_func = Pad(self.vocab_config['eos_id'])

        def _parse_batch(batch_ids):
            return pad_func([ids + [self.vocab_config['eos_id']] for ids in batch_ids])

        examples = []
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

    @serving
    def predict(self, data: List[str], batch_size: int = 1, n_best: int = 1, use_gpu: bool = False):

        if n_best > self.beam_size:
            raise ValueError(f'Predict arg "n_best" must be smaller or equal to self.beam_size, \
                but got {n_best} > {self.beam_size}')

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        batches = self._batchify(data, batch_size)

        results = []
        self.eval()
        for batch in batches:
            src_batch_ids = paddle.to_tensor(batch)
            trg_batch_beams = self(src_batch_ids).numpy().transpose([0, 2, 1])

            for trg_sample_beams in trg_batch_beams:
                for beam_idx, beam in enumerate(trg_sample_beams):
                    if beam_idx >= n_best:
                        break
                    trg_sample_ids = post_process_seq(beam, self.vocab_config['bos_id'], self.vocab_config['eos_id'])
                    trg_sample_words = self.trg_vocab.to_tokens(trg_sample_ids)
                    trg_sample_text = self.tokenizer.detokenize(trg_sample_words)
                    results.append(trg_sample_text)

        return results
