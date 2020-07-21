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

import cv2
import numpy as np
import paddlehub as hub


class EfficientNetB0SmallTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='efficientnetb0_small_imagenet')
        self.test_images = [
            "../image_dataset/classification/animals/dog.jpeg",
            "../image_dataset/keypoint_detection/girl2.jpg"
        ]
        self.true_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3).tolist()
        self.true_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3).tolist()

    def test_classifcation(self):
        results_1 = self.module.classify(paths=self.test_images, use_gpu=True)
        results_2 = self.module.classify(paths=self.test_images, use_gpu=False)
        for index, res in enumerate(results_1):
            self.assertTrue(res.keys(), results_2[index].keys())
            diff = list(res.values())[0] - list(results_2[index].values())[0]
            self.assertTrue((diff < 1e-5))

        test_images = [cv2.imread(img) for img in self.test_images]
        results_3 = self.module.classify(images=test_images, use_gpu=False)
        for index, res in enumerate(results_1):
            self.assertTrue(res.keys(), results_3[index].keys())

        results_4 = self.module.classify(
            images=test_images, use_gpu=True, top_k=2)
        for res in results_4:
            self.assertEqual(len(res.keys()), 2)

    def test_common_apis(self):
        width = self.module.get_expected_image_width()
        height = self.module.get_expected_image_height()
        mean = self.module.get_pretrained_images_mean()
        std = self.module.get_pretrained_images_std()

        self.assertEqual(width, 224)
        self.assertEqual(height, 224)
        self.assertEqual(mean.tolist(), self.true_mean)
        self.assertEqual(std.tolist(), self.true_std)


if __name__ == '__main__':
    main()
