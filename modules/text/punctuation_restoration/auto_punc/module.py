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
import re
from typing import List, Union

import numpy as np
import paddle
from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo, serving
from paddlehub.utils.log import logger
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.data import Pad


@moduleinfo(
    name="auto_punc",
    version="1.0.0",
    summary="",
    author="KPatrick",
    author_email="",
    type="text/punctuation_restoration")
class Ernie(paddle.nn.Layer):
    def __init__(self):
        super(Ernie, self).__init__()
        res_dir = os.path.join(MODULE_HOME, 'auto_punc')
        punc_vocab_file = os.path.join(res_dir, 'assets', 'punc_vocab.txt')
        ckpt_dir = os.path.join(res_dir, 'assets', 'ckpt')

        self.punc_vocab = self._load_dict(punc_vocab_file)
        self.punc_list = list(self.punc_vocab.keys())
        self.model = ErnieForTokenClassification.from_pretrained(ckpt_dir)
        self.model.eval()
        self.tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

    @staticmethod
    def _load_dict(dict_path):
        vocab = {}
        i = 0
        with open(dict_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                key = line.strip('\n')
                vocab[key] = i
                i += 1
        return vocab

    @staticmethod
    def _clean_text(text, punc_list):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
        text = re.sub(f'[{"".join([p for p in punc_list][1:])}]', '', text)
        return text

    def forward(self, text: str):
        wav = None
        input_ids = self.frontend.get_input_ids(text, merge_sentences=True)
        phone_ids = input_ids["phone_ids"]
        for part_phone_ids in phone_ids:
            with paddle.no_grad():
                mel = self.fastspeech2_inference(part_phone_ids)
                temp_wav = self.pwg_inference(mel)
                if wav is None:
                    wav = temp_wav
                else:
                    wav = paddle.concat([wav, temp_wav])
        return wav

    @serving
    def add_puncs(self, texts: Union[str, List[str]], max_length=256, device='cpu'):
        assert isinstance(texts, str) or (isinstance(texts, list) and isinstance(texts[0], str)), \
            'Input data should be str or List[str], but got {}'.format(type(texts))

        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        seg_ids = []
        seq_len = []
        for i in range(len(texts)):
            clean_text = self._clean_text(texts[i], self.punc_list)
            assert len(clean_text) > 0, f'Invalid input string: {texts[i]}'

            tokenized_input = self.tokenizer(
                list(clean_text), return_length=True, is_split_into_words=True, max_seq_len=max_length)

            input_ids.append(tokenized_input['input_ids'])
            seg_ids.append(tokenized_input['token_type_ids'])
            seq_len.append(tokenized_input['seq_len'])

        paddle.set_device(device)
        with paddle.no_grad():
            pad_func_for_input_ids = Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype='int64')
            pad_func_for_seg_ids = Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype='int64')
            input_ids = paddle.to_tensor(pad_func_for_input_ids(input_ids))
            seg_ids = paddle.to_tensor(pad_func_for_seg_ids(seg_ids))
            logits = self.model(input_ids, seg_ids)
            preds = paddle.argmax(logits, axis=-1)

        tokens = []
        labels = []
        for i in range(len(input_ids)):
            tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[i, 1:seq_len[i] - 1].tolist()))
            labels.append(preds[i, 1:seq_len[i] - 1].tolist())  # Remove predictions of special tokens.

        punc_texts = []
        for token, label in zip(tokens, labels):
            assert len(token) == len(label)
            text = ''
            for t, l in zip(token, label):
                text += t
                if l != 0:  # Non punc.
                    text += self.punc_list[l]
            punc_texts.append(text)

        return punc_texts
