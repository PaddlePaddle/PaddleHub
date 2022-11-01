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
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite('tests/test.jpg', img)
        cls.module = hub.Module(name="swinir_m_real_sr_x2")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('swinir_m_real_sr_x2_output')

    def test_real_sr1(self):
        results = self.module.real_sr(image='tests/test.jpg', visualization=False)

        self.assertIsInstance(results, np.ndarray)

    def test_real_sr2(self):
        results = self.module.real_sr(image=cv2.imread('tests/test.jpg'), visualization=True)

        self.assertIsInstance(results, np.ndarray)

    def test_real_sr3(self):
        results = self.module.real_sr(image=cv2.imread('tests/test.jpg'), visualization=True)

        self.assertIsInstance(results, np.ndarray)

    def test_real_sr4(self):
        self.assertRaises(Exception, self.module.real_sr, image=['tests/test.jpg'])

    def test_real_sr5(self):
        self.assertRaises(FileNotFoundError, self.module.real_sr, image='no.jpg')


if __name__ == "__main__":
    unittest.main()
