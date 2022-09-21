import os
import shutil
import unittest

import cv2
import requests
import numpy as np
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
        cls.module = hub.Module(name="lseg")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('lseg_output')

    def test_segment1(self):
        results = self.module.segment(
            image='tests/test.jpg',
            labels=['other', 'cat'],
            visualization=False
        )

        self.assertIsInstance(results['mix'], np.ndarray)
        self.assertIsInstance(results['color'], np.ndarray)
        self.assertIsInstance(results['gray'], np.ndarray)
        self.assertIsInstance(results['classes']['other'], np.ndarray)
        self.assertIsInstance(results['classes']['cat'], np.ndarray)

    def test_segment2(self):
        results = self.module.segment(
            image=cv2.imread('tests/test.jpg'),
            labels=['other', 'cat'],
            visualization=True
        )

        self.assertIsInstance(results['mix'], np.ndarray)
        self.assertIsInstance(results['color'], np.ndarray)
        self.assertIsInstance(results['gray'], np.ndarray)
        self.assertIsInstance(results['classes']['other'], np.ndarray)
        self.assertIsInstance(results['classes']['cat'], np.ndarray)

    def test_segment3(self):
        results = self.module.segment(
            image=cv2.imread('tests/test.jpg'),
            labels=['其他', '猫'],
            visualization=False
        )

        self.assertIsInstance(results['mix'], np.ndarray)
        self.assertIsInstance(results['color'], np.ndarray)
        self.assertIsInstance(results['gray'], np.ndarray)
        self.assertIsInstance(results['classes']['其他'], np.ndarray)
        self.assertIsInstance(results['classes']['猫'], np.ndarray)

    def test_segment4(self):
        self.assertRaises(
            Exception,
            self.module.segment,
            image=['tests/test.jpg'],
            labels=['other', 'cat']
        )

    def test_segment5(self):
        self.assertRaises(
            AttributeError,
            self.module.segment,
            image='no.jpg',
            labels=['other', 'cat']
        )


if __name__ == "__main__":
    unittest.main()
