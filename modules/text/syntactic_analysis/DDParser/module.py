# -*- coding:utf-8 -*-
import os
import argparse

import paddlehub as hub
from paddlehub.module.module import serving, moduleinfo, runnable
from paddlenlp import Taskflow


@moduleinfo(
    name="ddparser",
    version="1.1.0",
    summary="Baidu's open-source DDParser model.",
    author="baidu-nlp",
    author_email="",
    type="nlp/syntactic_analysis")
class ddparser(hub.NLPPredictionModule):
    def __init__(self,
                 tree=True,
                 prob=False, 
                 use_pos=False,
                 batch_size=1,
                 return_visual=False,
                 ):
        self.ddp = Taskflow(
            "dependency_parsing",
            tree=tree, 
            prob=prob, 
            use_pos=use_pos,
            batch_size=batch_size,
            return_visual=return_visual)

    @serving
    def serving_parse(self, texts):
        results = self.parse(texts)
        for i in range(len(results)):
            org_list = results[i]["head"]
            results[i]["head"] = [str(x) for x in org_list]
        return results

    def parse(self, texts):
        """
        parse the dependency.

        Args:
            texts(str or list[str]): the input texts to be parse.

        Returns:
            results(list[dict]): a list, with elements corresponding to each of the elements in texts. The element is a dictionary of shape:
                {
                    'word': list[str], the tokenized words.
                    'head': list[int], the head ids.
                    'deprel': list[str], the dependency relation.
                    'prob': list[float], the prediction probility of the dependency relation.
                    'postag': list[str], the POS tag. If the element of the texts is list, the key 'postag' will not return.
                    'visual' : numpy.ndarray: the dependency visualization. Use cv2.imshow to show or cv2.imwrite to save it. If return_visual=False, it will not return.
                }
       """
        return self.ddp(texts)

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

        results = self.parse(texts=input_data)

        return results

    def visualize(self, text):
        """
        Visualize the dependency.

        Args:
            text(str): input text.

        Returns:
            data(numpy.ndarray): a numpy array, use cv2.imshow to show it or cv2.imwrite to save it.
        """

        if isinstance(text, str):
            result = self.ddp(text)[0]['visual']
            return result
        else:
            raise TypeError(
                "Invalid inputs, input text should be str, but type of {} found!".format(type(text))
            )
