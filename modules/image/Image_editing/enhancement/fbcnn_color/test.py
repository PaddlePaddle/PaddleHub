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
        img_url = 'https://unsplash.com/photos/mJaD10XeD7w/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8M3x8Y2F0fGVufDB8fHx8MTY2MzczNDc3Mw&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="fbcnn_color")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('fbcnn_color_output')

    def test_artifacts_removal1(self):
        results = self.module.artifacts_removal(image='tests/test.jpg', quality_factor=None, visualization=False)

        self.assertIsInstance(results, np.ndarray)

    def test_artifacts_removal2(self):
        results = self.module.artifacts_removal(image=cv2.imread('tests/test.jpg'),
                                                quality_factor=None,
                                                visualization=True)

        self.assertIsInstance(results, np.ndarray)

    def test_artifacts_removal3(self):
        results = self.module.artifacts_removal(image=cv2.imread('tests/test.jpg'),
                                                quality_factor=0.5,
                                                visualization=True)

        self.assertIsInstance(results, np.ndarray)

    def test_artifacts_removal4(self):
        self.assertRaises(Exception, self.module.artifacts_removal, image=['tests/test.jpg'])

    def test_artifacts_removal5(self):
        self.assertRaises(FileNotFoundError, self.module.artifacts_removal, image='no.jpg')


if __name__ == "__main__":
    unittest.main()
