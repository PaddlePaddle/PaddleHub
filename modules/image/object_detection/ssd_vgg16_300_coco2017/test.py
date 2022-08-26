import os
import unittest

import cv2
import requests
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
        cls.module = hub.Module(name="ssd_vgg16_300_coco2017")

    def test_object_detection1(self):
        results = self.module.object_detection(
            paths=['tests/test.jpg']
        )
        bbox = results[0]['data'][0]
        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']
        self.assertEqual(label, 'cat')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(400 < left < 600)
        self.assertTrue(2800 < right < 3000)
        self.assertTrue(700 < top < 1000)
        self.assertTrue(3900 < bottom < 4100)

    def test_object_detection2(self):
        results = self.module.object_detection(
            images=[cv2.imread('tests/test.jpg')]
        )
        bbox = results[0]['data'][0]
        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']
        self.assertEqual(label, 'cat')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(400 < left < 600)
        self.assertTrue(2800 < right < 3000)
        self.assertTrue(700 < top < 1000)
        self.assertTrue(3900 < bottom < 4100)

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')
        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams.info'))


if __name__ == "__main__":
    unittest.main()
