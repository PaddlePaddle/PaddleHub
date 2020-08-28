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

imgpath = [
    '../image_dataset/super_resolution/BSD100_001.png',
    '../image_dataset/super_resolution/BSD100_002.png',
    '../image_dataset/super_resolution/BSD100_003.png',
]


class TestHumanSeg(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.sr_model = hub.Module(name="dcscn")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.sr_model = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_single_pic(self):
        with fluid.program_guard(self.test_prog):
            img = cv2.imread(imgpath[0])
            result = self.sr_model.super_resolution(
                images=[img], use_gpu=False, visualization=True)
            print(result[0]['data'])

    def test_ndarray(self):
        with fluid.program_guard(self.test_prog):

            for pic_path in imgpath:
                img = cv2.imread(pic_path)
                result = self.sr_model.super_resolution(
                    images=[img],
                    output_dir='test_dcscn_model_output',
                    use_gpu=False,
                    visualization=True)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.sr_model.save_inference_model(
                dirname='test_dcscn_model', combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestHumanSeg('test_single_pic'))
    suite.addTest(TestHumanSeg('test_ndarray'))
    suite.addTest(TestHumanSeg('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
