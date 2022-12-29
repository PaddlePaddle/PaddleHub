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
        cls.module = hub.Module(name="ch_pp-ocrv3")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('ocr_result')

    def test_recognize_text1(self):
        results = self.module.recognize_text(
            paths=['tests/test.jpg'],
            use_gpu=False,
            visualization=False,
        )
        self.assertEqual(results[0]['data'], [{
            'text': 'GIVE.',
            'confidence': 0.9509768486022949,
            'text_box_position': [[283, 162], [352, 162], [352, 202], [283, 202]]
        }, {
            'text': 'THANKS',
            'confidence': 0.9943074584007263,
            'text_box_position': [[261, 202], [376, 202], [376, 239], [261, 239]]
        }])

    def test_recognize_text2(self):
        results = self.module.recognize_text(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=False,
        )
        self.assertEqual(results[0]['data'], [{
            'text': 'GIVE.',
            'confidence': 0.9509768486022949,
            'text_box_position': [[283, 162], [352, 162], [352, 202], [283, 202]]
        }, {
            'text': 'THANKS',
            'confidence': 0.9943074584007263,
            'text_box_position': [[261, 202], [376, 202], [376, 239], [261, 239]]
        }])

    def test_recognize_text3(self):
        results = self.module.recognize_text(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=True,
            visualization=False,
        )
        self.assertEqual(results[0]['data'], [{
            'text': 'GIVE.',
            'confidence': 0.9509768486022949,
            'text_box_position': [[283, 162], [352, 162], [352, 202], [283, 202]]
        }, {
            'text': 'THANKS',
            'confidence': 0.9943074584007263,
            'text_box_position': [[261, 202], [376, 202], [376, 239], [261, 239]]
        }])

    def test_recognize_text4(self):
        results = self.module.recognize_text(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=True,
        )
        self.assertEqual(results[0]['data'], [{
            'text': 'GIVE.',
            'confidence': 0.9509768486022949,
            'text_box_position': [[283, 162], [352, 162], [352, 202], [283, 202]]
        }, {
            'text': 'THANKS',
            'confidence': 0.9943074584007263,
            'text_box_position': [[261, 202], [376, 202], [376, 239], [261, 239]]
        }])

    def test_recognize_text5(self):
        self.assertRaises(AttributeError, self.module.recognize_text, images=['tests/test.jpg'])

    def test_recognize_text6(self):
        self.assertRaises(AssertionError, self.module.recognize_text, paths=['no.jpg'])

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model/model.pdiparams'))

        self.assertTrue(os.path.exists('./inference/model/_text_detector_module.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model/_text_detector_module.pdiparams'))


if __name__ == "__main__":
    unittest.main()
