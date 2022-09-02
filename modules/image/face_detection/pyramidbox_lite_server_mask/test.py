import os
import unittest

import cv2
import requests
import paddlehub as hub


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://unsplash.com/photos/iFgRcqHznqg/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MXx8ZmFjZXxlbnwwfHx8fDE2NjE5ODAyMTc&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="pyramidbox_lite_server_mask")

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
        self.assertTrue(0 < left < 2000)
        self.assertTrue(0 < right < 2000)
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
        self.assertTrue(0 < left < 2000)
        self.assertTrue(0 < right < 2000)
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
        self.assertTrue(0 < left < 2000)
        self.assertTrue(0 < right < 2000)
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
        self.assertTrue(0 < left < 2000)
        self.assertTrue(0 < right < 2000)
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

        self.assertTrue(os.path.exists('./inference/model_mask_detector.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model_mask_detector.pdiparams'))
        self.assertTrue(os.path.exists('./inference/model_pyramidbox_lite.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model_pyramidbox_lite.pdiparams'))


if __name__ == "__main__":
    unittest.main()
