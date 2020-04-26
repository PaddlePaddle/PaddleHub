# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import cv2
import paddle.fluid as fluid
import paddlehub as hub

pic_dir = '../image_dataset/face_detection/'


class TestPyramidBoxLiteServer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.face_detector = hub.Module(name='pyramidbox_lite_server')

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.face_detector = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_single_pic(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            for pic_path in pics_path_list:
                result = self.face_detector.face_detection(
                    paths=[pic_path],
                    use_gpu=True,
                    visualization=True,
                    shrink=0.5,
                    confs_threshold=0.6)
                print(result)

    def test_ndarray(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            pics_ndarray = list()
            for pic_path in pics_path_list:
                im = cv2.imread(pic_path)
                result = self.face_detector.face_detection(
                    images=[im],
                    output_dir='ndarray_output',
                    shrink=1,
                    confs_threshold=0.6,
                    use_gpu=True,
                    visualization=True)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.face_detector.save_inference_model(
                dirname='pyramidbox_lite_server',
                model_filename='model',
                combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestPyramidBoxLiteServer('test_single_pic'))
    suite.addTest(TestPyramidBoxLiteServer('test_ndarray'))
    suite.addTest(TestPyramidBoxLiteServer('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
