import os

import numpy as np
from paddlehub.common.logger import logger


class Tokenizer(object):
    """Base tokenizer class.
    """

    def __init__(self):
        pass

    def tokenize(self, text):
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """Simple version FMM(Forward Maximun Matching) word tokenizer. This tokenizer can only
       be used in topic model demo, but not in real business application scenarios.

       Notes: This tokenizer can only recognize the words in the corresponding vocab file.
    """

    def __init__(self, vocab_path):
        super().__init__()
        self.__max_word_len = 0
        self.__vocab = set()
        self.__load_vocab(vocab_path)

    def tokenize(self, text):
        """Tokenize the input string `text`, and return the tokenize result.
        """
        text_len = len(text)
        result = []
        i = 0
        while i < text_len:
            word = found_word = ""
            # Deal with English characters.
            if self.__is_eng_char(text[i]):
                for j in range(i, text_len + 1):
                    if j < text_len and self.__is_eng_char(text[j]):
                        word += self.__tolower(text[j])
                    else:
                        # Forward matching by character granularity.
                        if word in self.__vocab:
                            result.append(word)
                        i = j - 1
                        break
            else:
                for j in range(i, min(i + self.__max_word_len, text_len)):
                    word += text[j]
                    if word in self.__vocab:
                        found_word = word
                if len(found_word) > 0:
                    result.append(found_word)
                    i += len(found_word) - 1
            i += 1
        return result

    def contains(self, word):
        """Check whether the word is in the vocabulary.
        """
        return word in self.__vocab

    def __load_vocab(self, vocab_path):
        """Load the word dictionary.
        """
        with open(vocab_path, 'r', encoding='utf-8') as fin:
            vocab_size = 0
            for line in fin.readlines():
                fields = line.strip().split('\t')
                assert len(fields) >= 2
                word = fields[1]
                self.__max_word_len = max(self.__max_word_len, len(word))
                self.__vocab.add(word)
                vocab_size += 1

    def __is_eng_char(self, c):
        """Check whether char c is an English character.
        """
        return (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z')

    def __tolower(self, c):
        """Return the lowercase character of the corresponding character, or return
           the original character if there is no corresponding lowercase character.
        """
        return c.lower()


class LACTokenizer(Tokenizer):
    def __init__(self, vocab_path, lac):
        super().__init__()
        self.__max_word_len = 0
        self.__vocab = set()
        self.__lac = lac
        self.__load_vocab(vocab_path)

    def __load_vocab(self, vocab_path):
        """Load the word dictionary.
                """
        with open(vocab_path, 'r', encoding='utf-8') as fin:
            vocab_size = 0
            for line in fin.readlines():
                fields = line.strip().split('\t')
                assert len(fields) >= 2
                word = fields[1]
                self.__max_word_len = max(self.__max_word_len, len(word))
                self.__vocab.add(word)
                vocab_size += 1

    def tokenize(self, text):
        results = self.__lac.lexical_analysis(texts=[text], use_gpu=False, batch_size=1, return_tag=True)
        # Change English words to lower case.
        # And just preserve the word in vocab.
        words = results[0]["word"]
        result = []
        for word in words:
            word = word.lower()
            if word in self.__vocab:
                result.append(word)
        return result

    def contains(self, word):
        """Check whether the word is in the vocabulary.
        """
        return word in self.__vocab
