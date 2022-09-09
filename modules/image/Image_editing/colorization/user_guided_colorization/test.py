import os
import shutil
import unittest

import cv2
import requests
import numpy as np
import paddlehub as hub


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://unsplash.com/photos/1sLIu1XKQrY/download?ixid=MnwxMjA3fDB8MXxhbGx8MTJ8fHx8fHwyfHwxNjYyMzQxNDUx&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="user_guided_colorization")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('colorization')

    def test_predict1(self):
        results = self.module.predict(
            images=['tests/test.jpg'],
            visualization=False
        )
        gray = results[0]['gray']
        hint = results[0]['hint']
        real = results[0]['real']
        fake_reg = results[0]['fake_reg']

        self.assertIsInstance(gray, np.ndarray)
        self.assertIsInstance(hint, np.ndarray)
        self.assertIsInstance(real, np.ndarray)
        self.assertIsInstance(fake_reg, np.ndarray)

    def test_predict2(self):
        results = self.module.predict(
            images=[cv2.imread('tests/test.jpg')],
            visualization=False
        )
        gray = results[0]['gray']
        hint = results[0]['hint']
        real = results[0]['real']
        fake_reg = results[0]['fake_reg']

        self.assertIsInstance(gray, np.ndarray)
        self.assertIsInstance(hint, np.ndarray)
        self.assertIsInstance(real, np.ndarray)
        self.assertIsInstance(fake_reg, np.ndarray)

    def test_predict3(self):
        results = self.module.predict(
            images=[cv2.imread('tests/test.jpg')],
            visualization=True
        )
        gray = results[0]['gray']
        hint = results[0]['hint']
        real = results[0]['real']
        fake_reg = results[0]['fake_reg']

        self.assertIsInstance(gray, np.ndarray)
        self.assertIsInstance(hint, np.ndarray)
        self.assertIsInstance(real, np.ndarray)
        self.assertIsInstance(fake_reg, np.ndarray)
    
    def test_predict4(self):
        self.assertRaises(
            IndexError,
            self.module.predict,
            images=['no.jpg'],
            visualization=False
        )

if __name__ == "__main__":
    unittest.main()
