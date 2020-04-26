# coding=utf-8
import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

image_dir = '../image_dataset/object_detection/'


class TestSSDVGG300(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        self.ssd = hub.Module(name="ssd_vgg16_300_coco2017")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests."""
        self.ssd = None

    def setUp(self):
        self.test_prog = fluid.Program()
        "Call setUp() to prepare environment\n"

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_context(self):
        with fluid.program_guard(self.test_prog):
            get_prediction = True
            inputs, outputs, program = self.ssd.context(
                pretrained=True,
                trainable=True,
                get_prediction=get_prediction)
            image = inputs["image"]
            im_size = inputs["im_size"]
            if get_prediction:
                bbox_out = outputs['bbox_out']
            else:
                body_features = outputs['body_features']

    def test_object_detection(self):
        with fluid.program_guard(self.test_prog):
            zebra = cv2.imread(os.path.join(image_dir, 'zebra.jpg')).astype('float32')
            zebras = [zebra, zebra]
            ## only paths
            print(
                self.ssd.object_detection(
                    paths=[os.path.join(image_dir, 'cat.jpg')]))
            ## only images
            print(self.ssd.object_detection(images=zebras))
            ## paths and images
            print(
                self.ssd.object_detection(
                    paths=[
                        os.path.join(image_dir, 'cat.jpg'),
                        os.path.join(image_dir, 'dog.jpg'),
                        os.path.join(image_dir, 'giraffe.jpg')
                    ],
                    images=zebras,
                    batch_size=2,
                    score_thresh=0.5))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestSSDVGG300('test_object_detection'))
    suite.addTest(TestSSDVGG300('test_context'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
