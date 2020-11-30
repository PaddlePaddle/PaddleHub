# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
'''This file is modified from https://github.com/huggingface/transformers'''

import collections
import os
import pickle
import unicodedata
from typing import Dict, List, Optional, Union, Tuple

from paddle.utils import try_import

from paddlehub.text.utils import load_vocab, is_whitespace, is_control, is_punctuation, whitespace_tokenize, is_chinese_char


class BasicTokenizer(object):
    '''Runs basic tokenization (punctuation splitting, lower casing, etc.).'''

    def __init__(self, do_lower_case: bool = True, never_split: List[str] = None, tokenize_chinese_chars: bool = True):
        ''' Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
            never_split: (`optional`) list of str
                List of token not to split.
            tokenize_chinese_chars: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        '''
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text: str, never_split: List[str] = None):
        ''' Basic Tokenization of a piece of text.
            Split on 'white spaces' only, for sub-word tokenization, see WordPieceTokenizer.
        Args:
            **never_split**: (`optional`) list of str
                List of token not to split.
        '''
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text: str):
        '''Strips accents from a piece of text.'''
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text: str, never_split: List[str] = None):
        '''Splits punctuation on a piece of text.'''
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str):
        '''Adds whitespace around any CJK character.'''
        output = []
        for char in text:
            if is_chinese_char(char):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _clean_text(self, text: str):
        '''Performs invalid character removal and whitespace cleanup on text.'''
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or is_control(char):
                continue
            if is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def encode(self):
        raise NotImplementedError('This tokenizer can only do tokenize(...), '
                                  'the ability to convert tokens to ids has not been implemented')

    def decode(self):
        raise NotImplementedError('This tokenizer can only do tokenize(...), '
                                  'the ability to convert ids to tokens has not been implemented')


class WordpieceTokenizer(object):
    '''Runs WordPiece tokenization.'''

    def __init__(self, vocab: List[str], unk_token: str, max_input_chars_per_word: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        '''Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = 'unaffable'
          output = ['un', '##aff', '##able']
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        '''

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
                    if start > 0:
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

    def encode(self):
        raise NotImplementedError('This tokenizer can only do tokenize(...), '
                                  'the ability to convert tokens to ids has not been implemented')

    def decode(self):
        raise NotImplementedError('This tokenizer can only do tokenize(...), '
                                  'the ability to convert ids to tokens has not been implemented')


class BertTokenizer(object):
    '''
    Constructs a BERT tokenizer. Based on WordPiece.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to do basic tokenization before WordPiece.
        never_split (:obj:`bool`, `optional`, defaults to :obj:`True`):
            List of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`string`, `optional`, defaults to '[UNK]'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to '[SEP]'):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to '[PAD]'):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to '[CLS]'):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to '[MASK]'):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see: https://github.com/huggingface/transformers/issues/328
    '''

    def __init__(
            self,
            vocab_file: str,
            do_lower_case: bool = True,
            do_basic_tokenize: bool = True,
            never_split: List[str] = None,
            unk_token: str = '[UNK]',
            sep_token: str = '[SEP]',
            pad_token: str = '[PAD]',
            cls_token: str = '[CLS]',
            mask_token: str = '[MASK]',
            tokenize_chinese_chars: bool = True,
    ):
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.do_lower_case = do_lower_case
        self.all_special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]

        if not os.path.isfile(vocab_file):
            raise ValueError('Can\'t find a vocabulary file at path \'{}\'.'.format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

        self.unk_token_id = self.convert_tokens_to_ids(self.unk_token)
        self.sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.pad_token_type_id = 0
        self.cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        self.mask_token_id = self.convert_tokens_to_ids(self.mask_token)
        self.all_special_ids = self.convert_tokens_to_ids(self.all_special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _convert_token_to_id(self, token):
        ''' Converts a token (str) in an id using the vocab. '''
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        '''Converts an index (integer) in a token (str) using the vocab.'''
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        ''' Converts a sequence of tokens (string) in a single string. '''
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def convert_tokens_to_ids(self, tokens):
        ''' Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        '''
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def convert_ids_to_tokens(self, ids: Union[int, List[int]],
                              skip_special_tokens: bool = False) -> Union[int, List[int]]:
        ''' Converts a single index or a sequence of indices (integers) in a token '
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.
            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        '''
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def tokenize(self, text: str):
        ''' Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).
            Take care of added tokens.
            Args:
                text (:obj:`string`): The sequence to be encoded.
        '''
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def build_inputs_with_special_tokens(self, token_ids_0: List[int],
                                         token_ids_1: Optional[List[int]] = None) -> List[int]:
        '''
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs` with the appropriate special tokens.
        '''
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def num_special_tokens_to_add(self, pair=False):
        '''
        Returns the number of added tokens when encoding a sequence with special tokens.
        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.
        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.
        Returns:
            Number of tokens added to sequences
        '''
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def get_special_tokens_mask(self,
                                token_ids_0: List[int],
                                token_ids_1: Optional[List[int]] = None,
                                already_has_special_tokens: bool = False) -> List[int]:
        '''
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        '''

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError('You should not supply a second sequence if the provided sequence of '
                                 'ids is already formated with special tokens for the model.')
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_segment_ids_from_sequences(self, token_ids_0: List[int],
                                          token_ids_1: Optional[List[int]] = None) -> List[int]:
        '''
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs` according to the given sequence(s).
        '''
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def clean_up_tokenization(self, out_string: str) -> str:
        ''' Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        '''
        out_string = (out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(
            ' \' ', '\'').replace(' n\'t', 'n\'t').replace(' \'m', '\'m').replace(' \'s', '\'s').replace(
                ' \'ve', '\'ve').replace(' \'re', '\'re'))
        return out_string

    def truncate_sequences(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            num_tokens_to_remove: int = 0,
            truncation_strategy: str = 'longest_first',
            stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
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

    def encode(self,
               text: Union[str, List[str], List[int]],
               text_pair: Optional[Union[str, List[str], List[int]]] = None,
               max_seq_len: Optional[int] = None,
               pad_to_max_seq_len: bool = True,
               truncation_strategy: str = 'longest_first',
               return_position_ids: bool = False,
               return_segment_ids: bool = True,
               return_input_mask: bool = False,
               return_length: bool = True,
               return_overflowing_tokens: bool = False,
               return_special_tokens_mask: bool = False):
        '''
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_seq_len`` is specified.
        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
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
            return_position_ids (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return tokens position ids (default True).
            return_segment_ids (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to return token type IDs.
            return_input_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to return the attention mask.
            return_length (:obj:`int`, defaults to :obj:`True`):
                If set the resulting dictionary will include the length of each encoded inputs
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
        Return:
            A Dictionary of shape::
                {
                    input_ids: list[int],
                    position_ids: list[int] if return_position_ids is True (default)
                    segment_ids: list[int] if return_segment_ids is True (default)
                    input_mask: list[int] if return_input_mask is True (default)
                    seq_len: int if return_length is True (default)
                    overflowing_tokens: list[int] if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if return_special_tokens_mask is True
                }
            With the fields:
            - ``input_ids``: list of token ids to be fed to a model
            - ``position_ids``: list of token position ids to be fed to a model
            - ``segment_ids``: list of token type ids to be fed to a model
            - ``input_mask``: list of indices specifying which tokens should be attended to by the model
            - ``length``: the input_ids length
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_seq_len`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        '''

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    'Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.')

        ids = get_input_ids(text)
        pair_ids = get_input_ids(text_pair) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair))
        if max_seq_len and total_len > max_seq_len:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_seq_len,
                truncation_strategy=truncation_strategy,
            )
            if return_overflowing_tokens:
                encoded_inputs['overflowing_tokens'] = overflowing_tokens
                encoded_inputs['num_truncated_tokens'] = total_len - max_seq_len

        # Add special tokens
        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        segment_ids = self.create_segment_ids_from_sequences(ids, pair_ids)

        # Build output dictionnary
        encoded_inputs['input_ids'] = sequence
        if return_segment_ids:
            encoded_inputs['segment_ids'] = segment_ids
        if return_special_tokens_mask:
            encoded_inputs['special_tokens_mask'] = self.get_special_tokens_mask(ids, pair_ids)
        if return_length:
            encoded_inputs['seq_len'] = len(encoded_inputs['input_ids'])

        # Check lengths
        assert max_seq_len is None or len(encoded_inputs['input_ids']) <= max_seq_len

        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs['input_ids']) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs['input_ids'])
            if return_input_mask:
                encoded_inputs['input_mask'] = [1] * len(encoded_inputs['input_ids']) + [0] * difference
            if return_segment_ids:
                encoded_inputs['segment_ids'] = (encoded_inputs['segment_ids'] + [self.pad_token_type_id] * difference)
            if return_special_tokens_mask:
                encoded_inputs['special_tokens_mask'] = encoded_inputs['special_tokens_mask'] + [1] * difference
            encoded_inputs['input_ids'] = encoded_inputs['input_ids'] + [self.pad_token_id] * difference
        else:
            if return_input_mask:
                encoded_inputs['input_mask'] = [1] * len(encoded_inputs['input_ids'])

        if return_position_ids:
            encoded_inputs['position_ids'] = list(range(len(encoded_inputs['input_ids'])))

        return encoded_inputs

    def decode(self,
               token_ids: Union[List[int], Dict],
               only_convert_to_tokens: bool = False,
               skip_pad_token: bool = False,
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True):
        '''
        Converts a sequence of ids (integer) to a string if only_convert_to_tokens is False or a list a sequence of tokens (str)
        when only_convert_to_tokens is True.
        Args:
            token_ids: list of tokenized input ids or dict containing a key called 'input_ids', can be obtained using the `encode` methods.
            only_convert_to_tokens:  if set to True, will only return a list a sequence of tokens (str). `paddlehub.dataset.base_nlp_dataset` will use this optional argument.
            skip_pad_token: if set to True, will replace pad tokens.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        '''
        if isinstance(token_ids, dict):
            token_ids = token_ids['input_ids']

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        tokens = []
        for token in filtered_tokens:
            if skip_pad_token and token == self.pad_token:
                continue
            tokens.append(token)
        if only_convert_to_tokens:
            return tokens

        if tokens:
            text = self.convert_tokens_to_string(tokens)
        else:
            text = ''

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text


class ErnieTinyTokenizer(BertTokenizer):
    def __init__(
            self,
            vocab_file: str,
            spm_path: str,
            word_dict_path: str,
            do_lower_case: bool = True,
            unk_token: str = '[UNK]',
            sep_token: str = '[SEP]',
            pad_token: str = '[PAD]',
            cls_token: str = '[CLS]',
            mask_token: str = '[MASK]',
    ):
        mod = try_import('sentencepiece')
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.do_lower_case = do_lower_case
        self.all_special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]

        if not os.path.isfile(vocab_file):
            raise ValueError('Can\'t find a vocabulary file at path \'{}\'.'.format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        # Here is the difference with BertTokenizer.
        self.dict = pickle.load(open(word_dict_path, 'rb'))
        self.sp_model = mod.SentencePieceProcessor()
        self.window_size = 5
        self.sp_model.Load(spm_path)

        self.unk_token_id = self.convert_tokens_to_ids(self.unk_token)
        self.sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.pad_token_type_id = 0
        self.cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        self.mask_token_id = self.convert_tokens_to_ids(self.mask_token)
        self.all_special_ids = self.convert_tokens_to_ids(self.all_special_tokens)

    def cut(self, chars: List[str]):
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

    def tokenize(self, text: str):
        text = [s for s in self.cut(text) if s != ' ']
        if self.do_lower_case:
            text = [s.lower() for s in text]
        text = ' '.join(text)
        tokens = self.sp_model.EncodeAsPieces(text)
        in_vocab_tokens = []
        for token in tokens:
            if token in self.vocab:
                in_vocab_tokens.append(token)
            else:
                in_vocab_tokens.append(self.unk_token)
        return in_vocab_tokens
