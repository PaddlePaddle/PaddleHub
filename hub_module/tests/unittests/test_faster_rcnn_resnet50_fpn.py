# coding=utf-8
import os
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

image_dir = '../image_dataset/object_detection/'


class TestFasterRCNNR50FPN(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests."""
        self.faster_rcnn_r50_fpn = hub.Module(
            name="faster_rcnn_resnet50_fpn_coco2017")
        # self.faster_rcnn_r50_fpn = hub.Module(directory='')

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests."""
        self.faster_rcnn_r50_fpn = None

    def setUp(self):
        self.test_prog = fluid.Program()
        "Call setUp() to prepare environment\n"

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_context(self):
        with fluid.program_guard(self.test_prog):
            inputs, outputs, program = self.faster_rcnn_r50_fpn.context(
                pretrained=False, trainable=True, phase='train')
            image = inputs['image']
            im_info = inputs['im_info']
            im_shape = inputs['im_shape']
            gt_class = inputs['gt_class']
            gt_bbox = inputs['gt_bbox']
            is_crowd = inputs['is_crowd']
            head_feat = outputs['head_feat']
            rpn_cls_loss = outputs['rpn_cls_loss']
            rpn_reg_loss = outputs['rpn_reg_loss']
            generate_proposal_labels = outputs['generate_proposal_labels']

    def test_object_detection(self):
        with fluid.program_guard(self.test_prog):
            zebra = cv2.imread(os.path.join(image_dir,
                                            'zebra.jpg')).astype('float32')
            zebras = [zebra, zebra]
            detection_results = self.faster_rcnn_r50_fpn.object_detection(
                paths=[
                    os.path.join(image_dir, 'cat.jpg'),
                    os.path.join(image_dir, 'dog.jpg'),
                    os.path.join(image_dir, 'giraffe.jpg')
                ],
                images=zebras,
                batch_size=2,
                use_gpu=False,
                score_thresh=0.5)
            print(detection_results)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestFasterRCNNR50FPN('test_object_detection'))
    suite.addTest(TestFasterRCNNR50FPN('test_context'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
