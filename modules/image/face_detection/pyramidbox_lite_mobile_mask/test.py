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
        img_url = 'https://ai-studio-static-online.cdn.bcebos.com/7799a8ccc5f6471b9d56fb6eff94f82a08b70ca2c7594d3f99877e366c0a2619'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="pyramidbox_lite_mobile_mask")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('detection_result')

    def test_face_detection1(self):
        results = self.module.face_detection(
            paths=['tests/test.jpg'],
            use_gpu=False,
            visualization=False
        )
        bbox = results[0]['data'][0]

        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']
        
        self.assertEqual(label, 'NO MASK')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(1000 < left < 4000)
        self.assertTrue(1000 < right < 4000)
        self.assertTrue(0 < top < 2000)
        self.assertTrue(0 < bottom < 2000)

    def test_face_detection2(self):
        results = self.module.face_detection(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=False
        )
        bbox = results[0]['data'][0]

        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']
        
        self.assertEqual(label, 'NO MASK')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(1000 < left < 4000)
        self.assertTrue(1000 < right < 4000)
        self.assertTrue(0 < top < 2000)
        self.assertTrue(0 < bottom < 2000)

    def test_face_detection3(self):
        results = self.module.face_detection(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=True
        )
        bbox = results[0]['data'][0]

        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']
        
        self.assertEqual(label, 'NO MASK')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(1000 < left < 4000)
        self.assertTrue(1000 < right < 4000)
        self.assertTrue(0 < top < 2000)
        self.assertTrue(0 < bottom < 2000)

    def test_face_detection4(self):
        results = self.module.face_detection(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=True,
            visualization=False
        )
        bbox = results[0]['data'][0]

        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']
        
        self.assertEqual(label, 'NO MASK')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(1000 < left < 4000)
        self.assertTrue(1000 < right < 4000)
        self.assertTrue(0 < top < 2000)
        self.assertTrue(0 < bottom < 2000)

    def test_face_detection5(self):
        self.assertRaises(
            AssertionError,
            self.module.face_detection,
            paths=['no.jpg']
        )

    def test_face_detection6(self):
        self.assertRaises(
            AttributeError,
            self.module.face_detection,
            images=['test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model/face_detector.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model/face_detector.pdiparams'))

        self.assertTrue(os.path.exists('./inference/model/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
