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

import cv2
import paddlehub as hub


class ChineseOCRDBCRNNTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='chinese_ocr_db_crnn_mobile')
        self.test_images = [
            "../image_dataset/text_recognition/11.jpg",
            "../image_dataset/text_recognition/test_image.jpg"
        ]

    def test_detect_text(self):
        results_1 = self.module.recognize_text(
            paths=self.test_images, use_gpu=True)
        results_2 = self.module.recognize_text(
            paths=self.test_images, use_gpu=False)

        test_images = [cv2.imread(img) for img in self.test_images]
        results_3 = self.module.recognize_text(
            images=test_images, use_gpu=False)
        for i, res in enumerate(results_1):
            self.assertEqual(res['save_path'], '')

            for j, item in enumerate(res['data']):
                self.assertEqual(item['confidence'],
                                 results_2[i]['data'][j]['confidence'])
                self.assertEqual(item['confidence'],
                                 results_3[i]['data'][j]['confidence'])
                self.assertEqual(item['text'], results_2[i]['data'][j]['text'])
                self.assertEqual(item['text'], results_3[i]['data'][j]['text'])
                self.assertEqual((item['text_box_position'] == results_2[i]
                                  ['data'][j]['text_box_position']), True)
                self.assertEqual((item['text_box_position'] == results_3[i]
                                  ['data'][j]['text_box_position']), True)


if __name__ == '__main__':
    main()
