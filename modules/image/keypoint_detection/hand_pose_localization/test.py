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
        img_url = 'https://unsplash.com/photos/8UAUuP97RlY/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjYxODQxMzI1&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="hand_pose_localization")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('output')

    def test_keypoint_detection1(self):
        results = self.module.keypoint_detection(
            paths=['tests/test.jpg'],
            visualization=False
        )
        kps = results[0]
        self.assertIsInstance(kps, list)

    def test_keypoint_detection2(self):
        results = self.module.keypoint_detection(
            images=[cv2.imread('tests/test.jpg')],
            visualization=False
        )
        kps = results[0]
        self.assertIsInstance(kps, list)

    def test_keypoint_detection3(self):
        results = self.module.keypoint_detection(
            images=[cv2.imread('tests/test.jpg')],
            visualization=True
        )
        kps = results[0]
        self.assertIsInstance(kps, list)

    def test_keypoint_detection4(self):
        self.module = hub.Module(name="hand_pose_localization", use_gpu=True)
        results = self.module.keypoint_detection(
            images=[cv2.imread('tests/test.jpg')],
            visualization=False
        )
        kps = results[0]
        self.assertIsInstance(kps, list)

    def test_keypoint_detection5(self):
        self.assertRaises(
            AssertionError,
            self.module.keypoint_detection,
            paths=['no.jpg']
        )

    def test_keypoint_detection6(self):
        self.assertRaises(
            AttributeError,
            self.module.keypoint_detection,
            images=['test.jpg']
        )


if __name__ == "__main__":
    unittest.main()
