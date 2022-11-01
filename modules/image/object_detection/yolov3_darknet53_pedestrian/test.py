import os
import shutil
import unittest

import cv2
import requests
import paddlehub as hub


class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://ai-studio-static-online.cdn.bcebos.com/15310014bf794c87a1e3b289d904ecae122aafe8c8fe47fd98634e79a8e4012f'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="yolov3_darknet53_pedestrian")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('yolov3_pedestrian_detect_output')

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

        self.assertEqual(label, 'pedestrian')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(0 < left < 1000)
        self.assertTrue(1000 < right < 3500)
        self.assertTrue(500 < top < 1500)
        self.assertTrue(1000 < bottom < 4500)

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

        self.assertEqual(label, 'pedestrian')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(0 < left < 1000)
        self.assertTrue(1000 < right < 3500)
        self.assertTrue(500 < top < 1500)
        self.assertTrue(1000 < bottom < 4500)

    def test_object_detection3(self):
        results = self.module.object_detection(
            images=[cv2.imread('tests/test.jpg')],
            visualization=False
        )
        bbox = results[0]['data'][0]
        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']

        self.assertEqual(label, 'pedestrian')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(0 < left < 1000)
        self.assertTrue(1000 < right < 3500)
        self.assertTrue(500 < top < 1500)
        self.assertTrue(1000 < bottom < 4500)

    def test_object_detection4(self):
        self.assertRaises(
            AssertionError,
            self.module.object_detection,
            paths=['no.jpg']
        )

    def test_object_detection5(self):
        self.assertRaises(
            AttributeError,
            self.module.object_detection,
            images=['test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()