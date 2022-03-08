# -*- coding:utf-8 -*-
import os
import argparse

import paddle
import paddlehub as hub
from paddlehub.module.module import serving, moduleinfo, runnable
from paddlenlp import Taskflow


@moduleinfo(
    name="nptag",
    version="1.0.0",
    summary="",
    author="Baidu",
    author_email="",
    type="nlp/text_to_knowledge",
    meta=hub.NLPPredictionModule)
class NPTag(paddle.nn.Layer):
    def __init__(self, 
                 batch_size=32, 
                 max_seq_length=128,
                 linking=True,
                 ):
        self.nptag = Taskflow("knowledge_mining", model="nptag", batch_size=batch_size, max_seq_length=max_seq_length, linking=linking)

    @serving
    def predict(self, texts):
        """
        The prediction interface for nptag.

        Args:
            texts(str or list[str]): the input texts to be predict.

        Returns:
            results(list[dict]): inference results. The element is a dictionary consists of:
                {
                    'text': str, the input texts.
                    'head': list[dict], tagging results, the element is a dictionary consists of:
                        {
                            'item': str, segmented word.
                            'offset': int, the offset compared with the first character.
                            'nptag_label':str, Part-Of-Speech label.
                            'length': int, word length.
                            'termid': str, link result with encyclopedia knowledge tree.
                        }
                }
        """
        return self.nptag(texts)

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")

        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        input_data = self.check_input_data(args)

        results = self.predict(texts=input_data)

        return results
