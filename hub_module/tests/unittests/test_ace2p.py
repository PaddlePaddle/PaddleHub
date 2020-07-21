# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

pic_dir = '../image_dataset/semantic_segmentation/'


class TestAce2p(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.human_parsing = hub.Module(name="ace2p")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.human_parsing = None

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
                result = self.human_parsing.segmentation(
                    paths=[pic_path], use_gpu=True, visualization=True)
                print(result)

    def test_batch(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            result = self.human_parsing.segmentation(
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
            for pic_path in pics_path_list:
                im = cv2.imread(pic_path)
                result = self.human_parsing.segmentation(
                    images=[im],
                    output_dir='ndarray_output',
                    use_gpu=True,
                    visualization=True)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.human_parsing.save_inference_model(
                dirname='ace2p', model_filename='model', combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestAce2p('test_single_pic'))
    suite.addTest(TestAce2p('test_batch'))
    suite.addTest(TestAce2p('test_ndarray'))
    suite.addTest(TestAce2p('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
