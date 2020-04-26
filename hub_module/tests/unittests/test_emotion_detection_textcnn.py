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

import paddlehub as hub


class EmotionDetectionTextCNNTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='emotion_detection_textcnn')
        self.test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
        self.results = [{
            'text': '这家餐厅很好吃',
            'emotion_label': 1,
            'emotion_key': 'neutral',
            'positive_probs': 0.3839,
            'negative_probs': 0.0414,
            'neutral_probs': 0.5747
        },
                        {
                            'text': '这部电影真的很差劲',
                            'emotion_label': 1,
                            'emotion_key': 'neutral',
                            'positive_probs': 0.3875,
                            'negative_probs': 0.1259,
                            'neutral_probs': 0.4866
                        }]
        self.labels = {"positive": 2, "negative": 0, "neutral": 1}

    def test_emotion_classify(self):
        # test batch_size
        results = self.module.emotion_classify(
            texts=self.test_text, use_gpu=False, batch_size=1)
        self.assertEqual(results, self.results)
        results = self.module.emotion_classify(
            texts=self.test_text, use_gpu=False, batch_size=10)
        self.assertEqual(results, self.results)

        # test gpu
        results = self.module.emotion_classify(
            texts=self.test_text, use_gpu=True, batch_size=1)
        self.assertEqual(results, self.results)

    def test_get_vocab_path(self):
        true_vocab_path = os.path.join(self.module.directory, "assets",
                                       "vocab.txt")
        vocab_path = self.module.get_vocab_path()
        self.assertEqual(vocab_path, true_vocab_path)

    def test_get_labels(self):
        labels = self.module.get_labels()
        self.assertEqual(labels, self.labels)


if __name__ == '__main__':
    main()
