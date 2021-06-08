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

import re
from typing import List

import codecs
from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt.apply_bpe import BPE


class MTTokenizer(object):
    def __init__(self, bpe_codes_file: str, lang_src: str = 'en', lang_trg: str = 'de', separator='@@'):
        self.moses_tokenizer = MosesTokenizer(lang=lang_src)
        self.moses_detokenizer = MosesDetokenizer(lang=lang_trg)
        self.bpe_tokenizer = BPE(
            codes=codecs.open(bpe_codes_file, encoding='utf-8'),
            merges=-1,
            separator=separator,
            vocab=None,
            glossaries=None)

    def tokenize(self, text: str):
        """
        Convert source string into bpe tokens.
        """
        moses_tokens = self.moses_tokenizer.tokenize(text)
        tokenized_text = ' '.join(moses_tokens)
        tokenized_bpe_text = self.bpe_tokenizer.process_line(tokenized_text)  # Apply bpe to text
        bpe_tokens = tokenized_bpe_text.split(' ')
        return bpe_tokens

    def detokenize(self, tokens: List[str]):
        """
        Convert target bpe tokens into string.
        """
        separator = self.bpe_tokenizer.separator
        text_with_separators = ' '.join(tokens)
        clean_text = re.sub(f'({separator} )|({separator} ?$)', '', text_with_separators)
        clean_tokens = clean_text.split(' ')
        detokenized_text = self.moses_detokenizer.tokenize(clean_tokens, return_str=True)
        return detokenized_text


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [int(idx) for idx in seq[:eos_pos + 1] if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)]
    return seq
