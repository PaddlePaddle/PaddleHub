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

import jieba
import paddle
from paddlenlp.transformers import position_encoding_init
from paddlenlp.transformers import WordEmbedding, PositionalEmbedding
from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo, serving

from transformer_nist_wait_3.model import SimultaneousTransformer
from transformer_nist_wait_3.processor import STACLTokenizer, predict


@moduleinfo(
    name="transformer_nist_wait_3",
    version="1.0.0",
    summary="",
    author="PaddlePaddle",
    author_email="",
    type="nlp/simultaneous_translation",
)
class STTransformer():
    """
    Transformer model for simultaneous translation.
    """

    # Model config
    model_config = {
        # Number of head used in multi-head attention.
        "n_head": 8,
        # Number of sub-layers to be stacked in the encoder and decoder.
        "n_layer": 6,
        # The dimension for word embeddings, which is also the last dimension of
        # the input and output of multi-head attention, position-wise feed-forward
        # networks, encoder and decoder.
        "d_model": 512,
    }

    def __init__(self, 
                 max_length=256,
                 max_out_len=256,
                 ):
        super(STTransformer, self).__init__()
        bpe_codes_fpath = os.path.join(MODULE_HOME, "transformer_nist_wait_3", "assets", "2M.zh2en.dict4bpe.zh")
        src_vocab_fpath = os.path.join(MODULE_HOME, "transformer_nist_wait_3", "assets", "nist.20k.zh.vocab")
        trg_vocab_fpath = os.path.join(MODULE_HOME, "transformer_nist_wait_3", "assets", "nist.10k.en.vocab")
        params_fpath = os.path.join(MODULE_HOME, "transformer_nist_wait_3", "assets", "transformer.pdparams")
        self.max_length = max_length
        self.max_out_len = max_out_len
        self.tokenizer = STACLTokenizer(
            bpe_codes_fpath,
            src_vocab_fpath,
            trg_vocab_fpath,
        )
        src_vocab_size = self.tokenizer.src_vocab_size
        trg_vocab_size = self.tokenizer.trg_vocab_size
        self.transformer = SimultaneousTransformer(
            src_vocab_size,
            trg_vocab_size,
            max_length=self.max_length,
            n_layer=self.model_config['n_layer'],
            n_head=self.model_config['n_head'],
            d_model=self.model_config['d_model'],
        )
        model_dict = paddle.load(params_fpath)
        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["src_pos_embedding.pos_encoder.weight"] = position_encoding_init(
            self.max_length + 1, self.model_config['d_model'])
        model_dict["trg_pos_embedding.pos_encoder.weight"] = position_encoding_init(
            self.max_length + 1, self.model_config['d_model'])
        self.transformer.load_dict(model_dict)

    @serving
    def translate(self, text, use_gpu=False):
        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        # Word segmentation
        text = ' '.join(jieba.cut(text))
        # For decoding max length
        decoder_max_length = 1
        # For decoding cache
        cache = None
        # For decoding start token id
        bos_id = None
        # Current source word index
        i = 0
        # For decoding: is_last=True, max_len=256
        is_last = False
        # Tokenized id
        user_input_tokenized = []
        # Store the translation
        result = []

        bpe_str, tokenized_src = self.tokenizer.tokenize(text)
        while i < len(tokenized_src):
            user_input_tokenized.append(tokenized_src[i])
            if bpe_str[i] in ['。', '？', '！']:
                is_last = True
            result, cache, bos_id = predict(
                user_input_tokenized, 
                decoder_max_length,
                is_last, 
                cache, 
                bos_id, 
                result,
                self.tokenizer, 
                self.transformer,
                max_out_len=self.max_out_len)
            i += 1    
        return " ".join(result)