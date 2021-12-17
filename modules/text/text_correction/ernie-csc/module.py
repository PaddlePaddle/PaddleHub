# -*- coding:utf-8 -*-
import os
import argparse

import paddle
import paddlehub as hub
from paddlehub.module.module import serving, moduleinfo, runnable
from paddlenlp import Taskflow


@moduleinfo(
    name="ernie-csc",
    version="1.0.0",
    summary="",
    author="Baidu",
    author_email="",
    type="nlp/text_correction",
    meta=hub.NLPPredictionModule)
class Ernie_CSC(paddle.nn.Layer):
    def __init__(self, 
                 batch_size=32):
        self.corrector = Taskflow("text_correction", batch_size=batch_size)

    @serving
    def predict(self, texts):
        """
        The prediction interface for ernie-csc.

        Args:
            texts(str or list[str]): the input texts to be predict.

        Returns:
            results(list[dict]): inference results. The element is a dictionary consists of:
                {
                    'source': str, the input texts.
                    'target': str, the predicted correct texts.
                    'errors': list[dict], detail information of errors, the element is a dictionary consists of:
                        {
                            'position': int, index of wrong charactor.
                            'correction': int, the origin charactor and the predicted correct charactor.
                        }
                }
        """
        return self.corrector(texts)

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
