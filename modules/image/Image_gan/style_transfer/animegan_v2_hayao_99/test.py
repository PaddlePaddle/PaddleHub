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
        img_url = 'https://unsplash.com/photos/mJaD10XeD7w/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8M3x8Y2F0fGVufDB8fHx8MTY2MzczNDc3Mw&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        img = cv2.imread('tests/test.jpg')
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        cv2.imwrite('tests/test.jpg', img)
        cls.module = hub.Module(name="animegan_v2_hayao_99")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('output')

    def test_style_transfer1(self):
        results = self.module.style_transfer(paths=['tests/test.jpg'])
        self.assertIsInstance(results[0], np.ndarray)

    def test_style_transfer2(self):
        results = self.module.style_transfer(paths=['tests/test.jpg'], visualization=True)
        self.assertIsInstance(results[0], np.ndarray)

    def test_style_transfer3(self):
        results = self.module.style_transfer(images=[cv2.imread('tests/test.jpg')])
        self.assertIsInstance(results[0], np.ndarray)

    def test_style_transfer4(self):
        results = self.module.style_transfer(images=[cv2.imread('tests/test.jpg')], visualization=True)
        self.assertIsInstance(results[0], np.ndarray)

    def test_style_transfer5(self):
        self.assertRaises(AssertionError, self.module.style_transfer, paths=['no.jpg'])

    def test_style_transfer6(self):
        self.assertRaises(cv2.error, self.module.style_transfer, images=['tests/test.jpg'])


if __name__ == "__main__":
    unittest.main()
