# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
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
#################################################################################
"""
本文件定义数据的结构
"""

from collections import Counter

import numpy as np

from parser.nets import nn
from parser.utils import utils
from parser.utils import vocab


class RawField(object):
    """filed的基类"""

    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        """repr"""
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        """预处理函数"""
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        """转换sequences"""
        return [self.preprocess(seq) for seq in sequences]


class Field(RawField):
    """Field"""

    def __init__(self,
                 name,
                 pad=None,
                 unk=None,
                 bos=None,
                 lower=False,
                 use_vocab=True,
                 tokenize=None,
                 fn=None):
        """init"""
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [
            token for token in [pad, unk, bos] if token is not None
        ]

    def __repr__(self):
        """repr"""
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += ", ".join(params)
        s += ")"

        return s

    @property
    def pad_index(self):
        """pad index"""
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        """unk index"""
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        """bos index"""
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    def preprocess(self, sequence):
        """预处理函数"""
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1, embed=None):
        """基于数据集建立vocab和embed对象"""
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(corpus, self.name)
        counter = Counter(
            token for seq in sequences for token in self.preprocess(seq))
        self.vocab = vocab.Vocab(counter, min_freq, self.specials,
                                 self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = np.zeros((len(self.vocab), embed.dim),
                                  dtype=np.float32)
            self.embed[self.vocab[tokens]] = embed.vectors
            self.embed /= np.std(self.embed, ddof=1)

    def transform(self, sequences):
        """转换sequences，如将word转化为id, 添加bos标签等"""
        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [self.vocab[seq] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        sequences = [np.array(seq) for seq in sequences]

        return sequences


class SubwordField(Field):
    """SubwordField"""

    def __init__(self, *args, **kwargs):
        """init"""
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super(SubwordField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1, embed=None):
        """基于数据集建立vocab和embed对象"""
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(corpus, self.name)
        counter = Counter(
            piece for seq in sequences for token in seq
            for piece in self.preprocess(token))
        self.vocab = vocab.Vocab(counter, min_freq, self.specials,
                                 self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)

            self.embed = np.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors

    def transform(self, sequences):
        """转换sequences，如将char转化为id, 添加bos标签等"""
        sequences = [[self.preprocess(token) for token in seq]
                     for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i] for i in token] for token in seq]
                         for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]

        sequences = [
            nn.pad_sequence([np.array(ids[:self.fix_len])
                             for ids in seq], self.pad_index, self.fix_len)
            for seq in sequences
        ]

        return sequences
