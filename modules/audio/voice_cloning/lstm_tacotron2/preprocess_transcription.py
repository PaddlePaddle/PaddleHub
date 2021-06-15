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

import argparse
from pathlib import Path
import pickle
import re

from parakeet.frontend import Vocab
import tqdm

zh_pattern = re.compile("[\u4e00-\u9fa5]")

_tones = {'<pad>', '<s>', '</s>', '0', '1', '2', '3', '4', '5'}

_pauses = {'%', '$'}

_initials = {
    'b',
    'p',
    'm',
    'f',
    'd',
    't',
    'n',
    'l',
    'g',
    'k',
    'h',
    'j',
    'q',
    'x',
    'zh',
    'ch',
    'sh',
    'r',
    'z',
    'c',
    's',
}

_finals = {
    'ii',
    'iii',
    'a',
    'o',
    'e',
    'ea',
    'ai',
    'ei',
    'ao',
    'ou',
    'an',
    'en',
    'ang',
    'eng',
    'er',
    'i',
    'ia',
    'io',
    'ie',
    'iai',
    'iao',
    'iou',
    'ian',
    'ien',
    'iang',
    'ieng',
    'u',
    'ua',
    'uo',
    'uai',
    'uei',
    'uan',
    'uen',
    'uang',
    'ueng',
    'v',
    've',
    'van',
    'ven',
    'veng',
}

_ernized_symbol = {'&r'}

_specials = {'<pad>', '<unk>', '<s>', '</s>'}

_phones = _initials | _finals | _ernized_symbol | _specials | _pauses

phone_pad_token = '<pad>'
tone_pad_token = '<pad>'
voc_phones = Vocab(sorted(list(_phones)))
voc_tones = Vocab(sorted(list(_tones)))


def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None


def ernized(syllable):
    return syllable[:2] != "er" and syllable[-2] == 'r'


def convert(syllable):
    # expansion of o -> uo
    syllable = re.sub(r"([bpmf])o$", r"\1uo", syllable)
    # syllable = syllable.replace("bo", "buo").replace("po", "puo").replace("mo", "muo").replace("fo", "fuo")
    # expansion for iong, ong
    syllable = syllable.replace("iong", "veng").replace("ong", "ueng")

    # expansion for ing, in
    syllable = syllable.replace("ing", "ieng").replace("in", "ien")

    # expansion for un, ui, iu
    syllable = syllable.replace("un", "uen").replace("ui", "uei").replace("iu", "iou")

    # rule for variants of i
    syllable = syllable.replace("zi", "zii").replace("ci", "cii").replace("si", "sii")\
        .replace("zhi", "zhiii").replace("chi", "chiii").replace("shi", "shiii")\
        .replace("ri", "riii")

    # rule for y preceding i, u
    syllable = syllable.replace("yi", "i").replace("yu", "v").replace("y", "i")

    # rule for w
    syllable = syllable.replace("wu", "u").replace("w", "u")

    # rule for v following j, q, x
    syllable = syllable.replace("ju", "jv").replace("qu", "qv").replace("xu", "xv")

    return syllable


def split_syllable(syllable: str):
    """Split a syllable in pinyin into a list of phones and a list of tones.
    Initials have no tone, represented by '0', while finals have tones from
    '1,2,3,4,5'.

    e.g.

    zhang -> ['zh', 'ang'], ['0', '1']
    """
    if syllable in _pauses:
        # syllable, tone
        return [syllable], ['0']

    tone = syllable[-1]
    syllable = convert(syllable[:-1])

    phones = []
    tones = []

    global _initials
    if syllable[:2] in _initials:
        phones.append(syllable[:2])
        tones.append('0')
        phones.append(syllable[2:])
        tones.append(tone)
    elif syllable[0] in _initials:
        phones.append(syllable[0])
        tones.append('0')
        phones.append(syllable[1:])
        tones.append(tone)
    else:
        phones.append(syllable)
        tones.append(tone)
    return phones, tones
