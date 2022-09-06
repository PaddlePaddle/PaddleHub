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
        cls.module = hub.Module(name="humanseg_server")

    def test_segment1(self):
        results = self.module.segment(
            paths=['tests/test.jpg'],
            use_gpu=False,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segment2(self):
        results = self.module.segment(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segment3(self):
        results = self.module.segment(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=False,
            visualization=True
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segment4(self):
        results = self.module.segment(
            images=[cv2.imread('tests/test.jpg')],
            use_gpu=True,
            visualization=False
        )
        self.assertIsInstance(results[0]['data'], np.ndarray)

    def test_segment5(self):
        self.assertRaises(
            AssertionError,
            self.module.segment,
            paths=['no.jpg']
        )

    def test_segment6(self):
        self.assertRaises(
            AttributeError,
            self.module.segment,
            images=['test.jpg']
        )

    def test_video_stream_segment1(self):
        img_matting, cur_gray, optflow_map = self.module.video_stream_segment(
            frame_org=cv2.imread('tests/test.jpg'),
            frame_id=1,
            prev_gray=None,
            prev_cfd=None,
            use_gpu=False
        )
        self.assertIsInstance(img_matting, np.ndarray)
        self.assertIsInstance(cur_gray, np.ndarray)
        self.assertIsInstance(optflow_map, np.ndarray)
        img_matting, cur_gray, optflow_map = self.module.video_stream_segment(
            frame_org=cv2.imread('tests/test.jpg'),
            frame_id=2,
            prev_gray=cur_gray,
            prev_cfd=optflow_map,
            use_gpu=False
        )
        self.assertIsInstance(img_matting, np.ndarray)
        self.assertIsInstance(cur_gray, np.ndarray)
        self.assertIsInstance(optflow_map, np.ndarray)

    def test_video_stream_segment2(self):
        img_matting, cur_gray, optflow_map = self.module.video_stream_segment(
            frame_org=cv2.imread('tests/test.jpg'),
            frame_id=1,
            prev_gray=None,
            prev_cfd=None,
            use_gpu=True
        )
        self.assertIsInstance(img_matting, np.ndarray)
        self.assertIsInstance(cur_gray, np.ndarray)
        self.assertIsInstance(optflow_map, np.ndarray)
        img_matting, cur_gray, optflow_map = self.module.video_stream_segment(
            frame_org=cv2.imread('tests/test.jpg'),
            frame_id=2,
            prev_gray=cur_gray,
            prev_cfd=optflow_map,
            use_gpu=True
        )
        self.assertIsInstance(img_matting, np.ndarray)
        self.assertIsInstance(cur_gray, np.ndarray)
        self.assertIsInstance(optflow_map, np.ndarray)

    def test_video_segment1(self):
        self.module.video_segment(
            video_path="tests/test.avi",
            use_gpu=False,
            save_dir='humanseg_lite_video_result'
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
