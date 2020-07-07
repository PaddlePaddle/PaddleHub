# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import paddlehub as hub
from paddlehub.module.module import serving, moduleinfo
import LAC
from paddle import fluid

from dependency_parser.parser import model as md
from dependency_parser.parser.config import Environment
from dependency_parser.parser.utils import corpus
from dependency_parser.parser.utils import data
from dependency_parser.parser.utils import field


@moduleinfo(
    name="dependency_parser",
    version="1.0.0",
    summary="",
    author="baidu-nlp",
    author_email="",
    type="nlp/syntactic_analysis")
class DependencyParser(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "params", "model")
        self.fields_path = os.path.join(self.directory, "assets", "fields")
        with fluid.dygraph.guard(fluid.CPUPlace()):
            self.model = md.load(self.pretrained_model_path)
        self.lac = LAC.LAC()
        self.env = Environment(self.fields_path)
        self.env.fields = self.env.fields._replace(PHEAD=field.Field('probs'))

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
             results(list): the word segmentation results
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

        lac_rs = self.lac.run(texts)
        predicts = corpus.Corpus.load_lac_rs(lac_rs, self.env.fields)
        dataset = data.TextDataset(predicts, [self.env.WORD, self.env.FEAT])
        # set the data loader
        with fluid.dygraph.guard(place):
            dataset.loader = data.batchify(
                dataset, batch_size, use_multiprocess=False)

            pred_arcs, pred_rels, pred_probs = md.epoch_predict(
                self.env, self.model, dataset.loader)
        predicts.arcs = pred_arcs
        predicts.rels = pred_rels
        predicts.probs = pred_probs
        results = str(predicts)

        return results

    def get_labels(self):
        """
        Get the labels which was used when pretraining
        Returns:
             self.labels(dict)
        """
        self.labels = {"positive": 1, "negative": 0}
        return self.labels


if __name__ == "__main__":
    module = DependencyParser()
    # Data to be predicted
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    results = module.parse(texts=test_text)
    print(results)
