# coding=utf-8
import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub


class TestSSDMobileNet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        self.ssd = hub.Module(name="ssd_mobilenet_v1_pascal")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests."""
        self.ssd = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_context(self):
        with fluid.program_guard(self.test_prog):
            image = fluid.layers.data(
                name='image', shape=[3, 300, 300], dtype='float32')
            inputs, outputs, program = self.ssd.context(
                input_image=image,
                pretrained=False,
                trainable=True,
                param_prefix='BaiDu')
            image = inputs["image"]
            head_features = outputs["body_feats"]

    def test_object_detection(self):
        with fluid.program_guard(self.test_prog):
            image_dir = '../image_dataset/pascal_voc/'
            airplane = cv2.imread(os.path.join(
                image_dir, 'airplane.jpg')).astype('float32')
            airplanes = np.array([airplane, airplane])
            detection_results = self.ssd.object_detection(
                paths=[
                    os.path.join(image_dir, 'bird.jpg'),
                    os.path.join(image_dir, 'bike.jpg'),
                    os.path.join(image_dir, 'cowboy.jpg'),
                    os.path.join(image_dir, 'sheep.jpg'),
                    os.path.join(image_dir, 'train.jpg')
                ],
                images=airplanes,
                batch_size=2)
            print(detection_results)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestSSDMobileNet('test_context'))
    suite.addTest(TestSSDMobileNet('test_object_detection'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
