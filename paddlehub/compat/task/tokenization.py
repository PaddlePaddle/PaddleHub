# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Tokenization classes.'''

import collections
import io
import pickle
import unicodedata
from typing import List, Union


def convert_to_unicode(text: Union[str, bytes]) -> str:
    '''Converts `text` to Unicode (if it's not already), assuming utf-8 input.'''
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError('Unsupported type: {}'.format(type(text)))


def load_vocab(vocab_file: str) -> List:
    '''Loads a vocabulary file into a dictionary.'''
    vocab = collections.OrderedDict()
    with io.open(vocab_file, 'r', encoding='UTF-8') as file:

        for num, line in enumerate(file):
            items = convert_to_unicode(line.strip()).split('\t')
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)

        return vocab


def convert_by_vocab(vocab: collections.OrderedDict, items: List[str]) -> List:
    '''Converts a sequence of [tokens|ids] using the vocab.'''
    output = []
    for item in items:
        output.append(vocab[item])

    return output


def convert_tokens_to_ids(vocab: collections.OrderedDict, tokens: List[str]) -> List:
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text: str) -> List:
    '''Runs basic whitespace cleaning and splitting on a peice of text.'''
    text = text.strip()
    if not text:
        return []

    tokens = text.split()
    return tokens


class FullTokenizer(object):
    '''Runs end-to-end tokenziation.'''

    def __init__(self, vocab_file: str, do_lower_case: bool = True, use_sentence_piece_vocab: bool = False):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.use_sentence_piece_vocab = use_sentence_piece_vocab
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, use_sentence_piece_vocab=self.use_sentence_piece_vocab)

    def tokenize(self, text: str) -> List:
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens: List) -> List:
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids: List) -> List:
        return convert_by_vocab(self.inv_vocab, ids)


class WSSPTokenizer(object):
    def __init__(self, vocab_file: str, sp_model_dir: str, word_dict: str, ws: bool = True, lower: bool = True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.ws = ws
        self.lower = lower
        self.dict = pickle.load(open(word_dict, 'rb'))

        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.window_size = 5
        self.sp_model.Load(sp_model_dir)

    def cut(self, chars: List) -> List:
        words = []
        idx = 0
        while idx < len(chars):
            matched = False
            for i in range(self.window_size, 0, -1):
                cand = chars[idx:idx + i]
                if cand in self.dict:
                    words.append(cand)
                    matched = True
                    break
            if not matched:
                i = 1
                words.append(chars[idx])
            idx += i
        return words

    def tokenize(self, text: Union[str, bytes], unk_token: str = '[UNK]') -> List:
        text = convert_to_unicode(text)
        if self.ws:
            text = [s for s in self.cut(text) if s != ' ']
        else:
            text = text.split(' ')
        if self.lower:
            text = [s.lower() for s in text]
        text = ' '.join(text)
        tokens = self.sp_model.EncodeAsPieces(text)
        in_vocab_tokens = []
        for token in tokens:
            if token in self.vocab:
                in_vocab_tokens.append(token)
            else:
                in_vocab_tokens.append(unk_token)
        return in_vocab_tokens

    def convert_tokens_to_ids(self, tokens: List) -> List:
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids: List) -> List:
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    '''Runs basic tokenization (punctuation splitting, lower casing, etc.).'''

    def __init__(self, do_lower_case: bool = True):
        '''Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        '''
        self.do_lower_case = do_lower_case

    def tokenize(self, text: Union[str, bytes]) -> List:
        '''Tokenizes a piece of text.'''
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text: str) -> str:
        '''Strips accents from a piece of text.'''
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text: str) -> List:
        '''Splits punctuation on a piece of text.'''
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        '''Adds whitespace around any CJK character.'''
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        '''Checks whether CP is the codepoint of a CJK character.'''
        # This defines a 'chinese character' as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text: str) -> str:
        '''Performs invalid character removal and whitespace cleanup on text.'''
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer(object):
    '''Runs WordPiece tokenziation.'''

    def __init__(self,
                 vocab: collections.OrderedDict,
                 unk_token: str = '[UNK]',
                 max_input_chars_per_word: int = 100,
                 use_sentence_piece_vocab: bool = False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.use_sentence_piece_vocab = use_sentence_piece_vocab

    def tokenize(self, text: Union[str, bytes]) -> List:
        '''Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
            input = 'unaffable'
            output = ['un', '##aff', '##able']
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        '''

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start == 0 and self.use_sentence_piece_vocab:
                        substr = u'\u2581' + substr
                    if start > 0 and not self.use_sentence_piece_vocab:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char: str) -> bool:
    '''Checks whether `chars` is a whitespace character.'''
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char: str) -> bool:
    '''Checks whether `chars` is a control character.'''
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char: str) -> bool:
    '''Checks whether `chars` is a punctuation character.'''
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as '^', '$', and '`' are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False
