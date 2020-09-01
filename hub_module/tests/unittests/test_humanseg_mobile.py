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

pic_dir = '../image_dataset/human_segmentation/image/'

imgpath = [
    '../image_dataset/human_segmentation/image/ache-adult-depression-expression-41253.jpg',
    '../image_dataset/human_segmentation/image/allergy-cold-disease-flu-41284.jpg',
    '../image_dataset/human_segmentation/image/bored-female-girl-people-41321.jpg',
    '../image_dataset/human_segmentation/image/colors-hairdresser-cutting-colorimetry-159780.jpg',
    '../image_dataset/human_segmentation/image/pexels-photo-206339.jpg'
]


class TestHumanSeg(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.human_seg = hub.Module(name="humanseg_mobile")

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

            img = cv2.imread(pics_path_list[0])
            result = self.human_seg.segment(
                images=[img], use_gpu=False, visualization=True)
            print(result[0]['data'])

    def test_batch(self):
        with fluid.program_guard(self.test_prog):
            result = self.human_seg.segment(
                paths=imgpath,
                batch_size=2,
                output_dir='batch_output_hrnet',
                use_gpu=False,
                visualization=True)
            print(result)

    def test_ndarray(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            pics_ndarray = list()
            for pic_path in pics_path_list:
                result = self.human_seg.segment(
                    images=[cv2.imread(pic_path)],
                    output_dir='ndarray_output_hrnet',
                    use_gpu=False,
                    visualization=True)

    def test_save_inference_model(self):
        with fluid.program_guard(self.test_prog):
            self.human_seg.save_inference_model(
                dirname='humanseg_mobile', combined=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestHumanSeg('test_single_pic'))
    suite.addTest(TestHumanSeg('test_batch'))
    suite.addTest(TestHumanSeg('test_ndarray'))
    suite.addTest(TestHumanSeg('test_save_inference_model'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
