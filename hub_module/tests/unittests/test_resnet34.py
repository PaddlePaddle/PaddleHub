# coding=utf-8
import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub


class TestResNet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        # self.mobilenet_v1 = hub.Module(name="mobilenet_v1")
        self.resnet = hub.Module(name='resnet34_v2_imagenet')

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests."""
        self.resnet = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_context(self):
        with fluid.program_guard(self.test_prog):
            image = fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32')
            inputs, outputs, program = self.resnet.context(
                input_image=image,
                pretrained=False,
                trainable=True,
                param_prefix='BaiDu')
            image = inputs["image"]
            body_feats = outputs['body_feats']

    def test_classification(self):
        with fluid.program_guard(self.test_prog):
            image_dir = "../image_dataset/pascal_voc/"
            airplane = cv2.imread(os.path.join(
                image_dir, 'airplane.jpg')).astype('float32')
            airplanes = np.array([airplane, airplane])
            print(airplanes)
            classification_results = self.resnet.classification(
                paths=[
                    os.path.join(image_dir, 'bird.jpg'),
                    os.path.join(image_dir, 'bike.jpg'),
                    os.path.join(image_dir, 'cowboy.jpg'),
                    os.path.join(image_dir, 'sheep.jpg'),
                    os.path.join(image_dir, 'train.jpg')
                ],
                images=airplanes,
                batch_size=2)
            print(classification_results)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestResNet('test_context'))
    suite.addTest(TestResNet('test_classification'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
