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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import paddlehub as hub


class BERTChnBaseTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='bert_chinese_L-12_H-768_A-12')
        self.test_text = [[
            '飞桨（PaddlePaddle）是国内开源产业级深度学习平台', 'PaddleHub是飞桨生态的预训练模型应用工具'
        ], ["飞浆PaddleHub"]]

    def test_get_embedding(self):
        # test batch_size
        results = self.module.get_embedding(
            texts=self.test_text, use_gpu=False, batch_size=1)
        results_2 = self.module.get_embedding(
            texts=self.test_text, use_gpu=False, batch_size=10)
        # 2 sample results
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results_2), 2)
        # sequence embedding and token embedding results per sample
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results_2[0]), 2)
        # sequence embedding shape
        self.assertEqual(results[0][0].shape, (768, ))
        self.assertEqual(results_2[0][0].shape, (768, ))
        # token embedding shape, max_seq_len is 512
        self.assertEqual(results[0][1].shape, (512, 768))
        self.assertEqual(results_2[0][1].shape, (512, 768))

        # test gpu
        results_3 = self.module.get_embedding(
            texts=self.test_text, use_gpu=True, batch_size=1)
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
        true_layers = [i for i in range(12)]
        self.assertEqual(layers, true_layers)

    def test_get_spm_path(self):
        self.assertEqual(self.module.get_spm_path(), None)

    def test_get_word_dict_path(self):
        self.assertEqual(self.module.get_word_dict_path(), None)

    def test_get_vocab_path(self):
        vocab_path = self.module.get_vocab_path()
        true_vocab_path = os.path.join(self.module.directory, "assets",
                                       "vocab.txt")
        self.assertEqual(vocab_path, true_vocab_path)


if __name__ == '__main__':
    main()
