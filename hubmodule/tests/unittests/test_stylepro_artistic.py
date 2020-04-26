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

content_dir = '../image_dataset/style_tranfer/content/'
style_dir = '../image_dataset/style_tranfer/style/'


class TestStyleProjection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.style_projection = hub.Module(name="stylepro_artistic")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.style_projection = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_single_style(self):
        with fluid.program_guard(self.test_prog):
            content_paths = [
                os.path.join(content_dir, f) for f in os.listdir(content_dir)
            ]
            style_paths = [
                os.path.join(style_dir, f) for f in os.listdir(style_dir)
            ]
            for style_path in style_paths:
                t1 = time.time()
                self.style_projection.style_transfer(
                    paths=[{
                        'content': content_paths[0],
                        'styles': [style_path]
                    }],
                    alpha=0.8,
                    use_gpu=True)
                t2 = time.time()
                print('\nCost time: {}'.format(t2 - t1))

    def test_multiple_styles(self):
        with fluid.program_guard(self.test_prog):
            content_path = os.path.join(content_dir, 'chicago.jpg')
            style_paths = [
                os.path.join(style_dir, f) for f in os.listdir(style_dir)
            ]
            for j in range(len(style_paths) - 1):
                res = self.style_projection.style_transfer(
                    paths=[{
                        'content': content_path,
                        'styles': [style_paths[j], style_paths[j + 1]],
                        'weights': [1, 2]
                    }],
                    alpha=0.8,
                    use_gpu=True,
                    visualization=True)
                print('#' * 100)
                print(res)
                print('#' * 100)

    def test_input_ndarray(self):
        with fluid.program_guard(self.test_prog):
            content_arr = cv2.imread(os.path.join(content_dir, 'chicago.jpg'))
            content_arr = cv2.cvtColor(content_arr, cv2.COLOR_BGR2RGB)
            style_arrs_BGR = [
                cv2.imread(os.path.join(style_dir, f))
                for f in os.listdir(style_dir)
            ]
            style_arrs_list = [
                cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) for arr in style_arrs_BGR
            ]
            for j in range(len(style_arrs_list) - 1):
                self.style_projection.style_transfer(
                    images=[{
                        'content':
                        content_arr,
                        'styles': [style_arrs_list[j], style_arrs_list[j + 1]]
                    }],
                    alpha=0.8,
                    use_gpu=True,
                    output_dir='transfer_out',
                    visualization=True)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.style_projection.save_inference_model(
                dirname='stylepro_artistic',
                model_filename='model',
                combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestStyleProjection('test_single_style'))
    suite.addTest(TestStyleProjection('test_multiple_styles'))
    suite.addTest(TestStyleProjection('test_input_ndarray'))
    suite.addTest(TestStyleProjection('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
