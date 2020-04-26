# coding=utf-8
import os
import unittest

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

image_dir = '../image_dataset/object_detection/vehicles/'


class TestYOLOv3DarkNet53Vehicles(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        self.yolov3_vehicles_detect = hub.Module(
            name="yolov3_darknet53_vehicles")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests."""
        self.yolov3_vehicles_detect = None

    def setUp(self):
        self.test_prog = fluid.Program()
        "Call setUp() to prepare environment\n"

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_context(self):
        with fluid.program_guard(self.test_prog):
            get_prediction = True
            inputs, outputs, program = self.yolov3_vehicles_detect.context(
                pretrained=True, trainable=True, get_prediction=get_prediction)

            image = inputs["image"]
            im_size = inputs["im_size"]
            if get_prediction:
                bbox_out = outputs['bbox_out']
            else:
                head_features = outputs['head_features']

    def test_object_detection(self):
        with fluid.program_guard(self.test_prog):
            paths = list()
            for file_path in os.listdir(image_dir):
                paths.append(os.path.join(image_dir, file_path))

            detection_results = self.yolov3_vehicles_detect.object_detection(
                paths=paths, batch_size=3, visualization=True)
            print(detection_results)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestYOLOv3DarkNet53Vehicles('test_object_detection'))
    suite.addTest(TestYOLOv3DarkNet53Vehicles('test_context'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
