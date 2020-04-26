# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import six

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.common.paddle_helper import add_vars_prefix
from paddlehub.module.module import moduleinfo, serving

from senta_bilstm.net import bilstm_net
from senta_bilstm.processor import load_vocab, preprocess, postprocess


@moduleinfo(
    name="senta_bilstm",
    version="1.1.0",
    summary="Baidu's open-source Sentiment Classification System.",
    author="baidu-nlp",
    author_email="",
    type="nlp/sentiment_analysis")
class SentaBiLSTM(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "infer_model")
        self.vocab_path = os.path.join(self.directory, "assets/vocab.txt")
        self.word_dict = load_vocab(self.vocab_path)
        self._word_seg_module = None

        self.predict = self.sentiment_classify

        self._set_config()

    @property
    def word_seg_module(self):
        """
        lac module
        """
        if not self._word_seg_module:
            self._word_seg_module = hub.Module(name="lac")
        return self._word_seg_module

    def context(self, trainable=False):
        """
        Get the input ,output and program of the pretrained senta_bilstm

        Args:
             trainable(bool): whether fine-tune the pretrained parameters of senta_bilstm or not

        Returns:
             inputs(dict): the input variables of senta_bilstm (words)
             outputs(dict): the output variables of senta_bilstm (the sentiment prediction results)
             main_program(Program): the main_program of lac with pretrained prameters
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1)
            data_name = data.name

            pred, fc = bilstm_net(data, 1256606)
            pred_name = pred.name
            fc_name = fc.name

            prefix_name = "@HUB_{}@".format(self.name)
            add_vars_prefix(program=main_program, prefix=prefix_name)

            for param in main_program.global_block().iter_parameters():
                param.trainable = trainable

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            # load the senta_lstm pretrained model
            def if_exist(var):
                return os.path.exists(
                    os.path.join(self.pretrained_model_path, var.name))

            fluid.io.load_vars(
                exe, self.pretrained_model_path, predicate=if_exist)

            inputs = {
                "words":
                main_program.global_block().vars[prefix_name + data_name]
            }
            outputs = {
                "class_probs":
                main_program.global_block().vars[prefix_name + pred_name],
                "sentence_feature":
                main_program.global_block().vars[prefix_name + fc_name]
            }

            return inputs, outputs, main_program

    @serving
    def sentiment_classify(self, texts=[], data={}, use_gpu=False,
                           batch_size=1):
        """
        Get the sentiment prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted, if texts not data
             data(dict): key must be 'text', value is the texts to be predicted, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch

        Returns:
             results(list): the word segmentation results
        """
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
        except:
            use_gpu = False

        if texts != [] and isinstance(texts, list) and data == {}:
            predicted_data = texts
        elif texts == [] and isinstance(data, dict) and isinstance(
                data.get('text', None), list) and data['text']:
            predicted_data = data["text"]
        else:
            raise ValueError(
                "The input data is inconsistent with expectations.")

        predicted_data = self.to_unicode(predicted_data)
        start_idx = 0
        iteration = int(math.ceil(len(predicted_data) / batch_size))
        results = []
        for i in range(iteration):
            if i < (iteration - 1):
                batch_data = predicted_data[start_idx:(start_idx + batch_size)]
            else:
                batch_data = predicted_data[start_idx:]

            start_idx = start_idx + batch_size
            processed_results = preprocess(self.word_seg_module, batch_data,
                                           self.word_dict, use_gpu, batch_size)
            tensor_words = self.texts2tensor(processed_results)

            if use_gpu:
                batch_out = self.gpu_predictor.run([tensor_words])
            else:
                batch_out = self.cpu_predictor.run([tensor_words])
            batch_result = postprocess(batch_out[0], processed_results)
            results += batch_result
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
    senta = SentaBiLSTM()
    # Data to be predicted
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    # execute predict and print the result
    input_dict = {"text": test_text}
    results = senta.sentiment_classify(data=input_dict, batch_size=3)
    for index, result in enumerate(results):
        if six.PY2:
            print(
                json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
    results = senta.sentiment_classify(texts=test_text)
    for index, result in enumerate(results):
        if six.PY2:
            print(
                json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
