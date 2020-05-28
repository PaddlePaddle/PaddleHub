# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os

from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub import TransformerModule
from paddlehub.module.module import moduleinfo, runnable, serving
from paddlehub.reader.tokenization import convert_to_unicode, FullTokenizer
from paddlehub.reader.batching import pad_batch_data
import numpy as np

from ernie_skep_sentiment_analysis.model.ernie import ErnieModel, ErnieConfig


@moduleinfo(
    name="ernie_skep_sentiment_analysis",
    version="1.0.0",
    summary=
    "SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis. Ernie_skep_sentiment_analysis module is initialize with enie_1.0_chn_large when pretraining. This module is finetuned on ChnSentiCorp dataset to do sentiment claasification. It can do sentiment analysis prediction directly, label as positive or negative.",
    author="baidu-nlp",
    author_email="",
    type="nlp/sentiment_analysis",
)
class ErnieSkepSentimentAnalysis(TransformerModule):
    """
    Ernie_skep_sentiment_analysis module is initialize with enie_1.0_chn_large when pretraining.
    This module is finetuned on ChnSentiCorp dataset to do sentiment claasification.
    It can do sentiment analysis prediction directly, label as positive or negative.
    """

    def _initialize(self):
        ernie_config_path = os.path.join(self.directory, "assets",
                                         "ernie_1.0_large_ch.config.json")
        self.ernie_config = ErnieConfig(ernie_config_path)
        self.MAX_SEQ_LEN = 512
        self.vocab_path = os.path.join(self.directory, "assets",
                                       "ernie_1.0_large_ch.vocab.txt")
        self.params_path = os.path.join(self.directory, "assets", "params")

        self.infer_model_path = os.path.join(self.directory, "assets",
                                             "inference_step_601")
        self.tokenizer = FullTokenizer(vocab_file=self.vocab_path)

        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.label_map = {0: 'negative', 1: 'positive'}

        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        model_file_path = os.path.join(self.infer_model_path, 'model')
        params_file_path = os.path.join(self.infer_model_path, 'params')

        config = AnalysisConfig(model_file_path, params_file_path)
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False

        if use_gpu:
            config.enable_use_gpu(8000, 0)
        else:
            config.disable_gpu()

        config.disable_glog_info()

        self.predictor = create_paddle_predictor(config)

    def net(self, input_ids, position_ids, segment_ids, input_mask):
        """
        create neural network.
        Args:
            input_ids (tensor): the word ids.
            position_ids (tensor): the position ids.
            segment_ids (tensor): the segment ids.
            input_mask (tensor): the padding mask.

        Returns:
            pooled_output (tensor):  sentence-level output for classification task.
            sequence_output (tensor): token-level output for sequence task.
        """
        ernie = ErnieModel(
            src_ids=input_ids,
            position_ids=position_ids,
            sentence_ids=segment_ids,
            input_mask=input_mask,
            config=self.ernie_config,
            use_fp16=False)

        pooled_output = ernie.get_pooled_output()
        sequence_output = ernie.get_sequence_output()
        return pooled_output, sequence_output

    def array2tensor(self, arr_data):
        """
        convert numpy array to PaddleTensor
        """
        tensor_data = PaddleTensor(arr_data)
        return tensor_data

    @serving
    def predict_sentiment(self, texts=[], use_gpu=False):
        """
        Get the sentiment label for the predicted texts. It will be classified as positive and negative.
        Args:
            texts (list(str)): the data to be predicted.
            use_gpu (bool): Whether to use gpu or not.
        Returns:
            res (list): The result of sentiment label and probabilties.
        """

        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        results = []
        for text in texts:
            feature = self._convert_text_to_feature(text)
            inputs = [self.array2tensor(ndarray) for ndarray in feature]
            output = self.predictor.run(inputs)
            probilities = np.array(output[0].data.float_data())
            label = self.label_map[np.argmax(probilities)]
            result = {
                'text': text,
                'sentiment_label': label,
                'positive_probs': probilities[1],
                'negative_probs': probilities[0]
            }
            results.append(result)

        return results

    def _convert_text_to_feature(self, text):
        """
        Convert the raw text to feature which is needed to run program (feed_vars).
        """
        text_a = convert_to_unicode(text)
        tokens_a = self.tokenizer.tokenize(text_a)
        max_seq_len = 512

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[0:(max_seq_len - 2)]

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        task_ids = [0] * len(token_ids)

        padded_token_ids, input_mask = pad_batch_data([token_ids],
                                                      max_seq_len=max_seq_len,
                                                      pad_idx=self.pad_id,
                                                      return_input_mask=True)
        padded_text_type_ids = pad_batch_data([text_type_ids],
                                              max_seq_len=max_seq_len,
                                              pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data([position_ids],
                                             max_seq_len=max_seq_len,
                                             pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data([task_ids],
                                         max_seq_len=max_seq_len,
                                         pad_idx=self.pad_id)

        feature = [
            padded_token_ids, padded_position_ids, padded_text_type_ids,
            input_mask, padded_task_ids
        ]
        return feature

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the %s module." % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)
        results = self.predict_sentiment(
            texts=[args.input_text], use_gpu=args.use_gpu)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_text', type=str, default=None, help="data to be predicted")


if __name__ == '__main__':
    test_module = ErnieSkepSentimentAnalysis()
    test_texts = ['你不是不聪明，而是不认真', '虽然小明很努力，但是他还是没有考100分']
    results = test_module.predict_sentiment(test_texts, use_gpu=False)
    print(results)
    test_module.context(max_seq_len=128)
    print(test_module.get_embedding(texts=[['你不是不聪明，而是不认真']]))
    print(test_module.get_params_layer())
