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


class ChineseTextDetectionDBTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='chinese_text_detection_db_mobile')
        self.test_images = [
            "../image_dataset/text_recognition/11.jpg",
            "../image_dataset/text_recognition/test_image.jpg"
        ]

    def test_detect_text(self):
        results_1 = self.module.detect_text(
            paths=self.test_images, use_gpu=True)
        results_2 = self.module.detect_text(
            paths=self.test_images, use_gpu=False)

        test_images = [cv2.imread(img) for img in self.test_images]
        results_3 = self.module.detect_text(images=test_images, use_gpu=False)
        for index, res in enumerate(results_1):
            self.assertEqual(res['save_path'], '')
            self.assertEqual(res['data'], results_2[index]['data'])
            self.assertEqual(res['data'], results_3[index]['data'])


if __name__ == '__main__':
    main()
