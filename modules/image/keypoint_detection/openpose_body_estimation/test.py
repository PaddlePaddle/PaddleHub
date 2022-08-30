import os
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
        cls.module = hub.Module(name="openpose_body_estimation")

    def test_predict1(self):
        results = self.module.predict(
            img='tests/test.jpg',
            visualization=False
        )
        kps = results['candidate'].tolist()
        self.assertIsInstance(kps, list)

    def test_predict2(self):
        results = self.module.predict(
            img=cv2.imread('tests/test.jpg'),
            visualization=False
        )
        kps = results['candidate'].tolist()
        self.assertIsInstance(kps, list)

    def test_predict3(self):
        results = self.module.predict(
            img=cv2.imread('tests/test.jpg'),
            visualization=True
        )
        kps = results['candidate'].tolist()
        self.assertIsInstance(kps, list)

    def test_predict4(self):
        self.assertRaises(
            AttributeError,
            self.module.predict,
            img='no.jpg'
        )

    def test_predict5(self):
        self.assertRaises(
            AttributeError,
            self.module.predict,
            img=['test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model/openpose_body_estimation.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model/openpose_body_estimation.pdiparams'))


if __name__ == "__main__":
    unittest.main()
