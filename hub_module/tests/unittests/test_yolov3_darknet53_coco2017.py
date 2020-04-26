# coding=utf-8
import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

image_dir = '../image_dataset/object_detection/'


class TestYoloV3DarkNet53(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        self.yolov3 = hub.Module(name="yolov3_darknet53_coco2017")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests."""
        self.yolov3 = None

    def setUp(self):
        self.test_prog = fluid.Program()
        "Call setUp() to prepare environment\n"

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_context(self):
        with fluid.program_guard(self.test_prog):
            get_prediction = True
            inputs, outputs, program = self.yolov3.context(
                pretrained=True,
                trainable=True,
                get_prediction=get_prediction)
            image = inputs["image"]
            im_size = inputs["im_size"]
            if get_prediction:
                bbox_out = outputs['bbox_out']
            else:
                head_features = outputs['outputs']

    def test_object_detection(self):
        with fluid.program_guard(self.test_prog):
            zebra = cv2.imread(os.path.join(image_dir,
                                            'zebra.jpg')).astype('float32')
            zebras = [zebra, zebra]
            detection_results = self.yolov3.object_detection(
                paths=[
                    os.path.join(image_dir, 'cat.jpg'),
                    os.path.join(image_dir, 'dog.jpg'),
                    os.path.join(image_dir, 'giraffe.jpg')
                ],
                images=zebras,
                batch_size=2)
            print(detection_results)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestYoloV3DarkNet53('test_object_detection'))
    suite.addTest(TestYoloV3DarkNet53('test_context'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
