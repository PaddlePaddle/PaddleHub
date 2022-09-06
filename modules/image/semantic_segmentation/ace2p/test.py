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
        img_url = 'https://unsplash.com/photos/pg_WCHWSdT8/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjYyNDM2ODI4&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        img = cv2.imread('tests/test.jpg')
        video = cv2.VideoWriter('tests/test.avi', fourcc,
                                20.0, tuple(img.shape[:2]))
        for i in range(40):
            video.write(img)
        video.release()
        cls.module = hub.Module(name="ace2p")

    def test_segmentation1(self):
        results = self.module.segmentation(
            paths=['tests/test.jpg'],
            use_gpu=False,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segmentation2(self):
        results = self.module.segmentation(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segmentation3(self):
        results = self.module.segmentation(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=True
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segmentation4(self):
        results = self.module.segmentation(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=True,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segmentation5(self):
        self.assertRaises(
            AssertionError,
            self.module.segmentation,
            paths=['no.jpg']
        )

    def test_segmentation6(self):
        self.assertRaises(
            AttributeError,
            self.module.segmentation,
            images=['test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
