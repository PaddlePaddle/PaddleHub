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


class TestPyramidBoxLiteServerMask(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.mask_detector = hub.Module(name="pyramidbox_lite_server_mask")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.mask_detector = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_single_pic(self):
        with fluid.program_guard(self.test_prog):
            paths_list = [os.path.join(pic_dir, f) for f in os.listdir(pic_dir)]
            print('\n')
            for pic_path in paths_list:
                print(pic_path)
                result = self.mask_detector.face_detection(
                    paths=[pic_path],
                    use_gpu=True,
                    visualization=True,
                    use_multi_scale=True,
                    shrink=0.5,
                    confs_threshold=0.6)
                print(result)

    def test_batch(self):
        with fluid.program_guard(self.test_prog):
            paths_list = [os.path.join(pic_dir, f) for f in os.listdir(pic_dir)]
            result = self.mask_detector.face_detection(
                paths=paths_list,
                batch_size=5,
                use_gpu=True,
                visualization=True,
                output_dir='batch_out',
                use_multi_scale=True,
                shrink=0.5,
                confs_threshold=0.6)
            print(result)

    def test_ndarray(self):
        with fluid.program_guard(self.test_prog):
            paths_list = [os.path.join(pic_dir, f) for f in os.listdir(pic_dir)]
            pics_ndarray = list()
            im_list = list()
            for pic_path in paths_list:
                im = cv2.imread(pic_path)
                im_list.append(im)
            result = self.mask_detector.face_detection(
                images=im_list,
                output_dir='ndarray_output',
                shrink=1,
                confs_threshold=0.6,
                use_gpu=True,
                visualization=True)
            print(result)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.mask_detector.save_inference_model(
                dirname='pyramidbox_lite_server_mask_model',
                model_filename='model',
                combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestPyramidBoxLiteServerMask('test_single_pic'))
    suite.addTest(TestPyramidBoxLiteServerMask('test_batch'))
    suite.addTest(TestPyramidBoxLiteServerMask('test_ndarray'))
    suite.addTest(TestPyramidBoxLiteServerMask('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
