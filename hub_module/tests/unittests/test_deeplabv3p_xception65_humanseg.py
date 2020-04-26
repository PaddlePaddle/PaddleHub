# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

pic_dir = '../image_dataset/keypoint_detection/'


class TestHumanSeg(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.human_seg = None

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
                result = self.human_seg.segmentation(
                    paths=[pic_path], use_gpu=True, visualization=True)
                print(result)

    def test_batch(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            result = self.human_seg.segmentation(
                paths=pics_path_list,
                batch_size=5,
                output_dir='batch_output',
                use_gpu=True,
                visualization=True)
            print(result)

    def test_ndarray(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            pics_ndarray = list()
            for pic_path in pics_path_list:
                result = self.human_seg.segmentation(
                    images=[cv2.imread(pic_path)],
                    output_dir='ndarray_output',
                    use_gpu=True,
                    visualization=True)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.human_seg.save_inference_model(
                dirname='deeplabv3p_xception65_humanseg',
                model_filename='model',
                combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestHumanSeg('test_single_pic'))
    suite.addTest(TestHumanSeg('test_batch'))
    suite.addTest(TestHumanSeg('test_ndarray'))
    suite.addTest(TestHumanSeg('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
