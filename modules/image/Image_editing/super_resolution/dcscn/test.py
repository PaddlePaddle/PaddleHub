import os
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
        cls.module = hub.Module(name="dcscn")

    def test_reconstruct1(self):
        results = self.module.reconstruct(
            paths=['tests/test.jpg'],
            use_gpu=False,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_reconstruct2(self):
        results = self.module.reconstruct(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_reconstruct3(self):
        results = self.module.reconstruct(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=True
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_reconstruct4(self):
        results = self.module.reconstruct(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=True,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_reconstruct5(self):
        self.assertRaises(
            AssertionError,
            self.module.reconstruct,
            paths=['no.jpg']
        )

    def test_reconstruct6(self):
        self.assertRaises(
            AttributeError,
            self.module.reconstruct,
            images=['test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
