import os
import shutil
import unittest

import cv2
import numpy as np
import requests

import paddlehub as hub

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestHubModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://replicate.delivery/mgxm/5a860a3d-90b8-4eb5-9428-68398c3326ee/23.png'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="unie")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('unie_output')

    def test_night_enhancement1(self):
        results = self.module.night_enhancement(image='tests/test.jpg', visualization=False)

        self.assertIsInstance(results, np.ndarray)

    def test_night_enhancement2(self):
        results = self.module.night_enhancement(image=cv2.imread('tests/test.jpg'), visualization=True)

        self.assertIsInstance(results, np.ndarray)

    def test_night_enhancement3(self):
        results = self.module.night_enhancement(image=cv2.imread('tests/test.jpg'), visualization=True)

        self.assertIsInstance(results, np.ndarray)

    def test_night_enhancement4(self):
        self.assertRaises(Exception, self.module.night_enhancement, image=['tests/test.jpg'])

    def test_night_enhancement5(self):
        self.assertRaises(FileNotFoundError, self.module.night_enhancement, image='no.jpg')


if __name__ == "__main__":
    unittest.main()
