# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import os
from unittest import TestCase, main
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import paddlehub as hub


class ErnieSkepSentimentAnalysisTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='ernie_skep_sentiment_analysis')
        self.test_text = [[
            '飞桨（PaddlePaddle）是国内开源产业级深度学习平台', 'PaddleHub是飞桨生态的预训练模型应用工具'
        ], ["飞浆PaddleHub"]]
        self.test_data = ['你不是不聪明，而是不认真', '虽然小明很努力，但是他还是没有考100分']
        self.results = [{
            'text': '你不是不聪明，而是不认真',
            'sentiment_label': 'negative',
            'positive_probs': 0.10738213360309601,
            'negative_probs': 0.8926178216934204
        },
                        {
                            'text': '虽然小明很努力，但是他还是没有考100分',
                            'sentiment_label': 'negative',
                            'positive_probs': 0.053915347903966904,
                            'negative_probs': 0.9460846185684204
                        }]

    def test_classify_sentiment(self):
        results_1 = self.module.classify_sentiment(
            self.test_data, use_gpu=False)
        results_2 = self.module.classify_sentiment(self.test_data, use_gpu=True)

        for index, res in enumerate(results_1):
            self.assertEqual(res['text'], self.results[index]['text'])
            self.assertEqual(res['sentiment_label'],
                             self.results[index]['sentiment_label'])
            self.assertTrue(
                abs(res['positive_probs'] -
                    self.results[index]['positive_probs']) < 1e-6)
            self.assertTrue(
                abs(res['negative_probs'] -
                    self.results[index]['negative_probs']) < 1e-6)

            self.assertEqual(res['text'], results_2[index]['text'])
            self.assertEqual(res['sentiment_label'],
                             results_2[index]['sentiment_label'])
            self.assertTrue(
                abs(res['positive_probs'] -
                    results_2[index]['positive_probs']) < 1e-6)
            self.assertTrue(
                abs(res['negative_probs'] -
                    results_2[index]['negative_probs']) < 1e-6)

    def test_get_embedding(self):
        # test batch_size
        max_seq_len = 128
        results = self.module.get_embedding(
            texts=self.test_text,
            use_gpu=False,
            batch_size=1,
            max_seq_len=max_seq_len)
        results_2 = self.module.get_embedding(
            texts=self.test_text,
            use_gpu=False,
            batch_size=10,
            max_seq_len=max_seq_len)
        # 2 sample results
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results_2), 2)
        # sequence embedding and token embedding results per sample
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results_2[0]), 2)
        # sequence embedding shape
        self.assertEqual(results[0][0].shape, (1024, ))
        self.assertEqual(results_2[0][0].shape, (1024, ))
        # token embedding shape
        self.assertEqual(results[0][1].shape, (max_seq_len, 1024))
        self.assertEqual(results_2[0][1].shape, (max_seq_len, 1024))

        # test gpu
        results_3 = self.module.get_embedding(
            texts=self.test_text,
            use_gpu=True,
            batch_size=1,
            max_seq_len=max_seq_len)
        diff = np.abs(results[0][0] - results_3[0][0])
        self.assertTrue((diff < 1e-6).all)
        diff = np.abs(results[0][1] - results_3[0][1])
        self.assertTrue((diff < 1e-6).all)
        diff = np.abs(results[1][0] - results_3[1][0])
        self.assertTrue((diff < 1e-6).all)
        diff = np.abs(results[1][1] - results_3[1][1])
        self.assertTrue((diff < 1e-6).all)

    def test_get_params_layer(self):
        self.module.context()
        layers = self.module.get_params_layer()
        layers = list(set(layers.values()))
        true_layers = [i for i in range(24)]
        self.assertEqual(layers, true_layers)

    def test_get_spm_path(self):
        self.assertEqual(self.module.get_spm_path(), None)

    def test_get_word_dict_path(self):
        self.assertEqual(self.module.get_word_dict_path(), None)

    def test_get_vocab_path(self):
        vocab_path = self.module.get_vocab_path()
        true_vocab_path = os.path.join(self.module.directory, "assets",
                                       "ernie_1.0_large_ch.vocab.txt")
        self.assertEqual(vocab_path, true_vocab_path)


if __name__ == '__main__':
    main()
