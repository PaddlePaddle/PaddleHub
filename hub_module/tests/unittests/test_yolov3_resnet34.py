# coding=utf-8
import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub


class TestYoloV3ResNet34(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        self.yolov3 = hub.Module(name="yolov3_resnet34_coco2017")

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
            image = fluid.layers.data(
                name='image', shape=[3, 608, 608], dtype='float32')
            inputs, outputs, program = self.yolov3.context(
                input_image=image,
                pretrained=False,
                trainable=True,
                param_prefix='BaiDu')
            image = inputs["image"]
            im_size = inputs["im_size"]

    def test_object_detection(self):
        with fluid.program_guard(self.test_prog):
            image_dir = '../image_dataset/'
            zebra = cv2.imread(os.path.join(image_dir,
                                            'zebra.jpg')).astype('float32')
            zebra = np.array([zebra, zebra])
            detection_results = self.yolov3.object_detection(
                paths=[
                    os.path.join(image_dir, 'cat.jpg'),
                    os.path.join(image_dir, 'dog.jpg'),
                    os.path.join(image_dir, 'giraffe.jpg')
                ],
                images=zebra,
                batch_size=2)
            print(detection_results)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestYoloV3ResNet34('test_object_detection'))
    suite.addTest(TestYoloV3ResNet34('test_context'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
