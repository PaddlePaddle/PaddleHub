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
        img_url = 'https://unsplash.com/photos/KTzZVDjUsXw/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzM3fHx0ZXh0fGVufDB8fHx8MTY2MzUxMTExMQ&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="ch_pp-ocrv3_det")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('detection_result')

    def test_detect_text1(self):
        results = self.module.detect_text(
            paths=['tests/test.jpg'],
            use_gpu=False,
            visualization=False,
        )
        self.assertEqual(
            results[0]['data'],
            [[[261, 202], [376, 202], [376, 239], [261, 239]], [[283, 162], [352, 162], [352, 202], [283, 202]]])

    def test_detect_text2(self):
        results = self.module.detect_text(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=False,
        )
        self.assertEqual(
            results[0]['data'],
            [[[261, 202], [376, 202], [376, 239], [261, 239]], [[283, 162], [352, 162], [352, 202], [283, 202]]])

    def test_detect_text3(self):
        results = self.module.detect_text(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=True,
            visualization=False,
        )
        self.assertEqual(
            results[0]['data'],
            [[[261, 202], [376, 202], [376, 239], [261, 239]], [[283, 162], [352, 162], [352, 202], [283, 202]]])

    def test_detect_text4(self):
        results = self.module.detect_text(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=True,
        )
        self.assertEqual(
            results[0]['data'],
            [[[261, 202], [376, 202], [376, 239], [261, 239]], [[283, 162], [352, 162], [352, 202], [283, 202]]])

    def test_detect_text5(self):
        self.assertRaises(AttributeError, self.module.detect_text, images=['tests/test.jpg'])

    def test_detect_text6(self):
        self.assertRaises(AssertionError, self.module.detect_text, paths=['no.jpg'])

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
