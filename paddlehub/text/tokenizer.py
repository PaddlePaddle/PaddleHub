# -*- coding:utf-8 -*-
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import collections
import inspect
import os
from typing import Callable, List, Union

import paddlehub as hub
from paddlehub.utils.log import logger
from paddlehub.text.bert_tokenizer import BasicTokenizer
from paddlehub.text.utils import load_vocab, whitespace_tokenize


class CustomTokenizer(object):
    '''
    Customtokenizer which will tokenize the input text as words or phases and convert the words (str) to an index (int) using the vocab.
    If you would like tokens, please use `hub.BertTokenizer`.
    '''

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 pad_token: str = '[PAD]',
                 tokenize_chinese_chars: bool = True,
                 cut_function: Callable = None):
        ''' Constructs a CustomTokenizer.
        Args:
            vocab_file (:obj:`string`): File containing the vocabulary.
            do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether to lower case the input if the input is in English
            pad_token (:obj:`string`, `optional`, defaults to '[PAD]'): The token used for padding, for example when batching sequences of different lengths.
            tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether to tokenize Chinese characters.
            cut_function(:obj:`function`): It is a function that aims to segment a chinese text and get the word segmentation result (list).
        '''

        if not os.path.isfile(vocab_file):
            raise ValueError('Can\'t find a vocabulary file at path \'{}\'.'.format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.pad_token = pad_token
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)

        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, tokenize_chinese_chars=tokenize_chinese_chars)

        self.cut_function = cut_function
        if not self.cut_function:
            lac = hub.Module(name='lac')
            self.cut_function = lac.cut
        elif inspect.isfunction(self.cut_function):
            self.cut_function = cut_function
        else:
            raise RuntimeError('The cut_function (%s) is not a true function.')

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _convert_token_to_id(self, token: str):
        ''' Converts a token (str) in an id using the vocab. '''
        return self.vocab.get(token, None)

    def _convert_id_to_token(self, index: int):
        '''Converts an index (integer) in a token (str) using the vocab.'''
        return self.ids_to_tokens.get(index, None)

    def convert_tokens_to_string(self, tokens: List[str]):
        ''' Converts a sequence of tokens (string) in a single string. '''
        if self.tokenize_chinese_chars:
            out_string = ''.join(tokens).strip()
        else:
            out_string = ' '.join(tokens).strip()
        return out_string

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_pad_token: bool):
        ''' Converts a single index or a sequence of indices (integers) in a token '
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.
            Args:
                ids(:obj:`int` or :obj:`List[int]`): list of tokenized input ids.
                skip_special_token: Don't decode special tokens (self.all_special_tokens). Default: False
        '''
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_pad_token and index == self.pad_token_id:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]):
        ''' Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        '''
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            wid = self._convert_token_to_id(token)
            if wid is not None:
                ids.append(wid)
        return ids

    def tokenize(self, text: str):
        '''
        Converts a string in a sequence of tokens (string), using the tokenizer.
        Text in chinese will be splitted in words using the Word Segmentor (Baidu_LAC) defaultly.
        If cut_function is set, it will be splitted in words using cut_function.
        Args:
            text (`string`): The sequence to be encoded.
        Returns:
            split_tokens (`list`): split
        '''
        if self.tokenize_chinese_chars:
            splitted_tokens = self.cut_function(text=text)
        else:
            splitted_tokens = self.basic_tokenizer.tokenize(text=text)
        return splitted_tokens

    def encode(self,
               text: str,
               text_pair: Union[str, List[str], List[int]] = None,
               max_seq_len: int = None,
               pad_to_max_seq_len: bool = True,
               truncation_strategy: str = 'longest_first',
               return_length: bool = True,
               return_overflowing_tokens: bool = False):
        '''
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_seq_len`` is specified.
        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                It's nonsense, just for compatible.
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            pad_to_max_seq_len (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set the resulting dictionary will include the length of each encoded inputs
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
        Return:
            A Dictionary of shape::
                {
                    text: list[int],
                    seq_len: int if return_length is True (default)
                    overflowing_tokens: list[int] if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                }
            With the fields:
            - ``text``: list of token ids to be fed to a model
            - ``length``: the input_ids length
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
        '''

        def get_input_ids(text: str):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                ids = self.convert_tokens_to_ids(tokens)
                return ids
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    'Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.')

        ids = get_input_ids(text)
        len_ids = len(ids)

        encoded_inputs = {}
        # When all words are not found in the vocab, it will return {}.
        if not len_ids:
            return encoded_inputs

        # Truncation: Handle max sequence length
        if max_seq_len and len_ids > max_seq_len:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids, num_tokens_to_remove=len_ids - max_seq_len, truncation_strategy=truncation_strategy)
            if return_overflowing_tokens:
                encoded_inputs['overflowing_tokens'] = overflowing_tokens
                encoded_inputs['num_truncated_tokens'] = len_ids - max_seq_len

        ## Check length and Pad
        if pad_to_max_seq_len and len(ids) < max_seq_len:
            encoded_inputs['text'] = ids + [self.pad_token_id] * (max_seq_len - len(ids))
        else:
            encoded_inputs['text'] = ids

        if return_length:
            encoded_inputs['seq_len'] = len(ids)

        return encoded_inputs

    def truncate_sequences(self,
                           ids: List[int],
                           pair_ids: List[int] = None,
                           num_tokens_to_remove: int = 0,
                           truncation_strategy: str = 'longest_first',
                           stride: int = 0):
        ''' Truncates a sequence pair in place to the maximum length.
        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_seq_len, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        '''
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError('Input sequence are too long for max_seq_len. Please select a truncation strategy.')
        else:
            raise ValueError(
                'Truncation_strategy should be selected in [\'longest_first\', \'only_first\', \'only_second\', \'do_not_truncate\']'
            )
        return (ids, pair_ids, overflowing_tokens)

    def decode(self,
               token_ids: List[int],
               only_convert_to_tokens: bool = True,
               skip_pad_token: bool = False,
               clean_up_tokenization_spaces: bool = True):
        '''
        Converts a sequence of ids (integer) to a string if only_convert_to_tokens is False or a list a sequence of tokens (str)
        when only_convert_to_tokens is True.
        Args:
            token_ids: list of tokenized input ids or dict with a key called 'text', can be obtained by using the `encode` methods.
            only_convert_to_tokens:  if set to True, will only return a list a sequence of tokens (str). `paddlehub.dataset.base_nlp_dataset` will use this optional argument.
            skip_pad_token: if set to True, will replace pad tokens.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        '''
        if isinstance(token_ids, dict):
            token_ids = token_ids['text']

        tokens = self.convert_ids_to_tokens(token_ids, skip_pad_token=skip_pad_token)

        if only_convert_to_tokens:
            return tokens

        if tokens and self.tokenize_chinese_chars:
            text = ''.join(self.convert_tokens_to_string(tokens))
        else:
            text = ' '.join(self.convert_tokens_to_string(tokens))

        if not self.tokenize_chinese_chars and clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def clean_up_tokenization(self, out_string: str) -> str:
        '''
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        '''
        out_string = (out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(
            ' \' ', '\'').replace(' n\'t', 'n\'t').replace(' \'m', '\'m').replace(' \'s', '\'s').replace(
                ' \'ve', '\'ve').replace(' \'re', '\'re'))
        return out_string
