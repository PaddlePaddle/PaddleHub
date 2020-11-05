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
from paddlehub.common.paddle_helper import get_variable_info
from paddlehub.module.module import moduleinfo, serving
from paddlehub.reader import tokenization

from porn_detection_gru.processor import load_vocab, preprocess, postprocess


@moduleinfo(
    name="porn_detection_gru",
    version="1.1.0",
    summary="Baidu's open-source Porn Detection Model.",
    author="baidu-nlp",
    author_email="",
    type="nlp/sentiment_analysis")
class PornDetectionGRU(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "infer_model")
        self.tokenizer_vocab_path = os.path.join(self.directory, "assets", "vocab.txt")
        self.vocab_path = os.path.join(self.directory, "assets", "word_dict.txt")
        self.vocab = load_vocab(self.vocab_path)
        self.sequence_max_len = 256
        self.tokenizer = tokenization.FullTokenizer(self.tokenizer_vocab_path)

        self.param_file = os.path.join(self.directory, "assets", "params.txt")

        self.predict = self.detection

        self._set_config()

    def context(self, trainable=False):
        """
        Get the input ,output and program of the pretrained porn_detection_gru
        Args:
             trainable(bool): whether fine-tune the pretrained parameters of porn_detection_gru or not
        Returns:
             inputs(dict): the input variables of porn_detection_gru (words)
             outputs(dict): the output variables of porn_detection_gru (the sentiment prediction results)
             main_program(Program): the main_program of lac with pretrained prameters
        """
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        program, feed_target_names, fetch_targets = fluid.io.load_inference_model(
            dirname=self.pretrained_model_path, executor=exe)

        with open(self.param_file, 'r') as file:
            params_list = file.readlines()
        for param in params_list:
            param = param.strip()
            var = program.global_block().var(param)
            var_info = get_variable_info(var)
            program.global_block().create_parameter(
                shape=var_info['shape'], dtype=var_info['dtype'], name=var_info['name'])

        for param in program.global_block().iter_parameters():
            param.trainable = trainable

        for name, var in program.global_block().vars.items():
            if name == feed_target_names[0]:
                inputs = {"words": var}
            # output of sencond layer from the end prediction layer (fc-softmax)
            if name == "@HUB_porn_detection_gru@layer_norm_0.tmp_2":
                outputs = {"class_probs": fetch_targets[0], "sentence_feature": var}
        return inputs, outputs, program

    @serving
    def detection(self, texts=[], data={}, use_gpu=False, batch_size=1):
        """
        Get the porn prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted, if texts not data
             data(dict): key must be 'text', value is the texts to be predicted, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch

        Returns:
             results(list): the porn prediction results
        """
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
        except:
            use_gpu = False

        if texts != [] and isinstance(texts, list) and data == {}:
            predicted_data = texts
        elif texts == [] and isinstance(data, dict) and isinstance(data.get('text', None), list) and data['text']:
            predicted_data = data["text"]
        else:
            raise ValueError("The input data is inconsistent with expectations.")

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
            processed_results = preprocess(batch_data, self.tokenizer, self.vocab, self.sequence_max_len)
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
        self.labels = {"porn": 1, "not_porn": 0}
        return self.labels


if __name__ == "__main__":
    porn_detection_gru = PornDetectionGRU()
    porn_detection_gru.context()
    # porn_detection_gru = hub.Module(name='porn_detection_gru')
    test_text = ["黄片下载", "打击黄牛党"]

    results = porn_detection_gru.detection(texts=test_text)
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
    input_dict = {"text": test_text}
    results = porn_detection_gru.detection(data=input_dict)
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
