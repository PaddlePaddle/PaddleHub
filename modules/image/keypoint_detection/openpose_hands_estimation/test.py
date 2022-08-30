import os
import unittest

import cv2
import requests
import paddlehub as hub


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://unsplash.com/photos/QUVLQPt37n0/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8M3x8cGVyc29uJTIwaGFuZHMlMjBoZWxsb3xlbnwwfHx8fDE2NjE4NjE1MzE&force=true&w=640'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="openpose_hands_estimation")

    def test_predict1(self):
        results = self.module.predict(
            img='tests/test.jpg',
            visualization=False
        )
        kps = results['all_hand_peaks'][0].tolist()
        self.assertIsInstance(kps, list)

    def test_predict2(self):
        results = self.module.predict(
            img=cv2.imread('tests/test.jpg'),
            visualization=False
        )
        kps = results['all_hand_peaks'][0].tolist()
        self.assertIsInstance(kps, list)

    def test_predict3(self):
        results = self.module.predict(
            img=cv2.imread('tests/test.jpg'),
            visualization=True
        )
        kps = results['all_hand_peaks'][0].tolist()
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

        self.assertTrue(os.path.exists('./inference/model/openpose_hands_estimation.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model/openpose_hands_estimation.pdiparams'))


if __name__ == "__main__":
    unittest.main()