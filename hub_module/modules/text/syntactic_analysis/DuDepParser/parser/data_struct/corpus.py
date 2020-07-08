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

from collections import namedtuple
from collections.abc import Iterable

from DuDepParser.parser.data_struct import Field

CoNLL = namedtuple(
    typename='CoNLL',
    field_names=[
        'ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL',
        'PHEAD', 'PDEPREL'
    ],
    defaults=[None] * 10)


class Sentence(object):
    """Sentence"""

    def __init__(self, fields, values):
        """init"""
        for field, value in zip(fields, values):
            if isinstance(field, Iterable):
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        return len(next(iter(self.values)))

    def __repr__(self):
        """repr"""
        return '\n'.join('\t'.join(map(str, line))
                         for line in zip(*self.values)) + '\n'

    def json(self):
        j = {}
        for field in self.fields:
            if isinstance(field, Iterable) and not field[0].name.isdigit():
                j[field[0].name] = getattr(self, field[0].name)
            elif not field.name.isdigit():
                j[field.name] = getattr(self, field.name)
        return j


class Corpus(object):
    """Corpus"""

    def __init__(self, fields, sentences):
        """init"""
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        """repr"""
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)

    def __setattr__(self, name, value):
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod
    def load(cls, path, fields):
        start, sentences = 0, []
        fields = [
            fd if fd is not None else Field(str(i))
            for i, fd in enumerate(fields)
        ]
        with open(path, 'r') as f:
            lines = [
                line.strip() for line in f if not line.startswith('#') and (
                    len(line) == 1 or line.split()[0].isdigit())
            ]
        for i, line in enumerate(lines):
            if not line:
                values = list(zip(*[j.split('\t') for j in lines[start:i]]))
                sentences.append(Sentence(fields, values))
                start = i + 1

        return cls(fields, sentences)

    @classmethod
    def load_lac_results(cls, inputs, fields):
        sentences = []
        fields = [
            fd if fd is not None else Field(str(i))
            for i, fd in enumerate(fields)
        ]
        for _input in inputs:
            if isinstance(_input[0], list):
                tokens, poss = _input
            else:
                tokens = _input
                poss = ['-'] * len(tokens)
            values = (
                list(range(1,
                           len(tokens) + 1)),
                tokens,
                tokens,
                poss,
                poss,
                *[['-'] * len(tokens) for _ in range(5)],
            )
            sentences.append(Sentence(fields, values))
        return cls(fields, sentences)

    @classmethod
    def load_word_segments(cls, inputs, fields):
        fields = [
            fd if fd is not None else Field(str(i))
            for i, fd in enumerate(fields)
        ]
        sentences = []
        for tokens in inputs:
            values = (
                list(range(1,
                           len(tokens) + 1)),
                tokens,
                tokens,
                *[['-'] * len(tokens) for _ in range(7)],
            )
            sentences.append(Sentence(fields, values))
        return cls(fields, sentences)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")

    def print(self):
        """print self"""
        print(self)

    def json(self):
        result = []
        for sentence in self:
            result.append(sentence.json())
        return result
