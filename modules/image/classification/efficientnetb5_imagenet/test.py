import os
import shutil
import unittest

import cv2
import requests

import paddlehub as hub

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestHubModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://unsplash.com/photos/brFsZ7qszSY/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8OHx8ZG9nfGVufDB8fHx8MTY2MzA1ODQ1MQ&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="efficientnetb5_imagenet")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')

    def test_classification1(self):
        results = self.module.classification(paths=['tests/test.jpg'])
        data = results[0]
        self.assertTrue('Pembroke' in data)
        self.assertTrue(data['Pembroke'] > 0.5)

    def test_classification2(self):
        results = self.module.classification(images=[cv2.imread('tests/test.jpg')])
        data = results[0]
        self.assertTrue('Pembroke' in data)
        self.assertTrue(data['Pembroke'] > 0.5)

    def test_classification3(self):
        results = self.module.classification(images=[cv2.imread('tests/test.jpg')], use_gpu=True)
        data = results[0]
        self.assertTrue('Pembroke' in data)
        self.assertTrue(data['Pembroke'] > 0.5)

    def test_classification4(self):
        self.assertRaises(AssertionError, self.module.classification, paths=['no.jpg'])

    def test_classification5(self):
        self.assertRaises(TypeError, self.module.classification, images=['tests/test.jpg'])

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
