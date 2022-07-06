# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import serving


@moduleinfo(
    name="jieba_paddle",
    version="1.0.1",
    summary=
    "jieba_paddle is a chineses tokenizer using BiGRU base on the PaddlePaddle deeplearning framework. More information please refer to https://github.com/fxsjy/jieba.",
    author="baidu-paddle",
    author_email="paddle-dev@gmail.com",
    type="nlp/lexical_analysis")
class JiebaPaddle(hub.Module):

    def _initialize(self):
        pass

    @serving
    def cut(self, sentence, use_paddle=True, cut_all=False, HMM=True):
        """
        The main function that segments an entire sentence that contains
        Chinese characters into separated words.
        Args:
            sentence(str): The str(unicode) to be segmented.
            use_paddle(bool): Whether use jieba paddle model or not. Default as true.
            cut_all(bool): Model type. True for full pattern, False for accurate pattern.
            HMM(bool): Whether to use the Hidden Markov Model.

        Returns:
            results(dict): The word segmentation result of the input sentence, whose key is 'word'.
        """
        self.check_dependency()
        import jieba
        jieba.setLogLevel(logging.ERROR)
        jieba._compat.setLogLevel(logging.ERROR)

        if use_paddle:
            jieba.enable_paddle()
            res = " ".join(jieba.cut(sentence, use_paddle=True))
            seg_list = res.strip(" ").split(" ")
        else:
            res = " ".join(jieba.cut(sentence, cut_all=cut_all, HMM=HMM))
            seg_list = res.strip(" ").split(" ")

        return seg_list

    def check_dependency(self):
        """
        Check jieba tool dependency.
        """
        try:
            import jieba
        except ImportError:
            print(
                'This module requires jieba tools. The running enviroment does not meet the requirments. Please install jieba packages.'
            )
            exit()

    def cut_for_search(self, sentence, HMM=True):
        """
        Finer segmentation for search engines.
        Args:
            sentence(str): The str(unicode) to be segmented.
            HMM(bool): Whether to use the Hidden Markov Model.

        Returns:
            results(dict): The word segmentation result of the input sentence, whose key is 'word'.
        """
        self.check_dependency()
        import jieba
        jieba.setLogLevel(logging.ERROR)
        res = " ".join(jieba.cut_for_search(sentence, HMM=HMM))
        seg_list = res.strip(" ").split(" ")
        return seg_list

    def load_userdict(self, user_dict):
        '''
        Load personalized dict to improve detect rate.
        Args:
            user_dict(str): A plain text file path. It contains words and their ocurrences. Can be a file-like object, or the path of the dictionary file,
            whose encoding must be utf-8.
                Structure of dict file:
                    word1 freq1 word_type1
                    word2 freq2 word_type2
                    ...

                Word type may be ignored
        '''
        self.check_dependency()
        import jieba
        jieba.setLogLevel(logging.ERROR)
        jieba.load_userdict("userdict.txt")

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Args:
            topK(int): return how many top keywords. `None` for all possible words.
            withWeight(bool): if True, return a list of (word, weight);
                          if False, return a list of words.
            allowPOS(tuple): the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            withFlag(bool): only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        Returns:
            result(list): The key words.
        """
        self.check_dependency()
        import jieba
        import jieba.analyse
        jieba.setLogLevel(logging.ERROR)
        res = jieba.analyse.extract_tags(sentence,
                                         topK=topK,
                                         withWeight=withWeight,
                                         allowPOS=allowPOS,
                                         withFlag=withFlag)
        return res

    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Args:
            topK(int): return how many top keywords. `None` for all possible words.
            withWeight(bool): if True, return a list of (word, weight);
                          if False, return a list of words.
            allowPOS(tuple): the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            withFlag(bool): only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        Returns:
            result(list): The key words.
        """
        self.check_dependency()
        import jieba
        jieba.setLogLevel(logging.ERROR)
        res = jieba.analyse.textrank(sentence, topK=topK, withWeight=withWeight, allowPOS=allowPOS, withFlag=withFlag)
        return res
