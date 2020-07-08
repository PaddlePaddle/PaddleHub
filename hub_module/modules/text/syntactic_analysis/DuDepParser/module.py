# -*- coding:utf-8 -*-
import os

import LAC
from paddle import fluid
import paddlehub as hub
from paddlehub.module.module import serving, moduleinfo

from DuDepParser.parser import epoch_predict
from DuDepParser.parser import load
from DuDepParser.parser import ArgConfig
from DuDepParser.parser import Environment
from DuDepParser.parser.data_struct import Corpus
from DuDepParser.parser.data_struct import TextDataset
from DuDepParser.parser.data_struct import batchify
from DuDepParser.parser.data_struct import Field


@moduleinfo(
    name="DuDepParser",
    version="1.0.0",
    summary="Baidu's open-source DuDepParser model based on char feature.",
    author="baidu-nlp",
    author_email="nlp@baidu.com",
    type="nlp/syntactic_analysis")
class DuDepParser(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets")
        self.config_path = os.path.join(self.directory, "assets", "config.ini")

        args = [
            f"--model_files={self.pretrained_model_path}",
            f"--config_path={self.config_path}",
            "--tree",
            "--prob",
        ]
        args = ArgConfig(args)
        args.log_path = None

        self.env = Environment(args)
        self.args = self.env.args

        with fluid.dygraph.guard(fluid.CPUPlace()):
            self.model = load(self.args.model_path)
        self.lac = LAC.LAC(mode='lac')
        self.use_pos = True
        self.env.fields = self.env.fields._replace(PHEAD=Field('prob'))
        self.env.fields = self.env.fields._replace(CPOS=Field('postag'))

        self.predict = self.parse

    @serving
    def parse(self, texts=[], use_gpu=False, batch_size=1):
        """
        Get the sentiment prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch

        Returns:
             results(list): the results.

        Example:
        >>> dpp = DuDepParser()
        >>> texts = ["百度是一家高科技公司"]
        >>> dpp.parse(texts)
        [{'word': ['百度', '是', '一家', '高科技', '公司'], 'postag': ['ORG', 'v', 'm', 'n', 'n'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'], 'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]
        """

        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        if not texts:
            return
        if isinstance(texts, str):
            texts = [texts]
        if all([isinstance(i, str) and i for i in texts]):
            lac_results = self.lac.run(texts)
            predicts = Corpus.load_lac_results(lac_results, self.env.fields)
        else:
            raise RuntimeError("please check the foramt of your inputs.")
        dataset = TextDataset(predicts, [self.env.WORD, self.env.FEAT])

        # set the data loader
        with fluid.dygraph.guard(place):
            dataset.loader = batchify(
                dataset,
                batch_size,
                use_multiprocess=False,
                sequential_sampler=True)
            pred_arcs, pred_rels, pred_probs = epoch_predict(
                self.env, self.args, self.model, dataset.loader)
            predicts.head = pred_arcs
            predicts.deprel = pred_rels
            predicts.prob = pred_probs

        return predicts.json()


if __name__ == "__main__":
    module = DuDepParser()
    # Data to be predicted
    test_text = ["百度是一家高科技公司"]
    results = module.parse(texts=test_text)
    print(results)
