import os
import shutil
import unittest

import cv2
import requests
import numpy as np
import paddlehub as hub


class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://ai-studio-static-online.cdn.bcebos.com/68313e182f5e4ad9907e69dac9ece8fc50840d7ffbd24fa88396f009958f969a'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="stgan_bald")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('bald_output')

    def test_bald1(self):
        results = self.module.bald(
            paths=['tests/test.jpg']
        )
        data_0 = results[0]['data_0']
        data_1 = results[0]['data_1']
        data_2 = results[0]['data_2']
        self.assertIsInstance(data_0, np.ndarray)
        self.assertIsInstance(data_1, np.ndarray)
        self.assertIsInstance(data_2, np.ndarray)

    def test_bald2(self):
        results = self.module.bald(
            images=[cv2.imread('tests/test.jpg')]
        )
        data_0 = results[0]['data_0']
        data_1 = results[0]['data_1']
        data_2 = results[0]['data_2']
        self.assertIsInstance(data_0, np.ndarray)
        self.assertIsInstance(data_1, np.ndarray)
        self.assertIsInstance(data_2, np.ndarray)

    def test_bald3(self):
        results = self.module.bald(
            images=[cv2.imread('tests/test.jpg')],
            visualization=False
        )
        data_0 = results[0]['data_0']
        data_1 = results[0]['data_1']
        data_2 = results[0]['data_2']
        self.assertIsInstance(data_0, np.ndarray)
        self.assertIsInstance(data_1, np.ndarray)
        self.assertIsInstance(data_2, np.ndarray)

    def test_bald4(self):
        self.assertRaises(
            AssertionError,
            self.module.bald,
            paths=['no.jpg']
        )

    def test_bald5(self):
        self.assertRaises(
            cv2.error,
            self.module.bald,
            images=['tests/test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
