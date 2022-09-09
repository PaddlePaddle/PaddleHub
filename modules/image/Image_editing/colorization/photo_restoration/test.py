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
        cls.module = hub.Module(name="photo_restoration")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('photo_restoration')

    def test_run_image1(self):
        results = self.module.run_image(
            input='tests/test.jpg'
        )
        self.assertIsInstance(results, np.ndarray)

    def test_run_image2(self):
        results = self.module.run_image(
            input=cv2.imread('tests/test.jpg')
        )
        self.assertIsInstance(results, np.ndarray)

    def test_run_image3(self):
        self.assertRaises(
            FileNotFoundError,
            self.module.run_image,
            input='no.jpg'
        )


if __name__ == "__main__":
    unittest.main()
