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
        img_url = 'https://ai-studio-static-online.cdn.bcebos.com/7799a8ccc5f6471b9d56fb6eff94f82a08b70ca2c7594d3f99877e366c0a2619'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="solov2")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('solov2_result')

    def test_predict1(self):
        results = self.module.predict(image='tests/test.jpg', visualization=False)
        segm = results['segm']
        label = results['label']
        score = results['score']
        self.assertIsInstance(segm, np.ndarray)
        self.assertIsInstance(label, np.ndarray)
        self.assertIsInstance(score, np.ndarray)

    def test_predict2(self):
        results = self.module.predict(image=cv2.imread('tests/test.jpg'), visualization=False)
        segm = results['segm']
        label = results['label']
        score = results['score']
        self.assertIsInstance(segm, np.ndarray)
        self.assertIsInstance(label, np.ndarray)
        self.assertIsInstance(score, np.ndarray)

    def test_predict3(self):
        results = self.module.predict(image=cv2.imread('tests/test.jpg'), visualization=True)
        segm = results['segm']
        label = results['label']
        score = results['score']
        self.assertIsInstance(segm, np.ndarray)
        self.assertIsInstance(label, np.ndarray)
        self.assertIsInstance(score, np.ndarray)

    def test_predict4(self):
        module = hub.Module(name="solov2", use_gpu=True)
        results = module.predict(image=cv2.imread('tests/test.jpg'), visualization=True)
        segm = results['segm']
        label = results['label']
        score = results['score']
        self.assertIsInstance(segm, np.ndarray)
        self.assertIsInstance(label, np.ndarray)
        self.assertIsInstance(score, np.ndarray)

    def test_predict5(self):
        self.assertRaises(FileNotFoundError, self.module.predict, image='no.jpg')

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
