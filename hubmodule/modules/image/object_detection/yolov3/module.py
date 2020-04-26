# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.common.paddle_helper import add_vars_prefix

from yolov3.data_feed import reader
from yolov3.processor import load_label_info, postprocess
from yolov3.yolo_head import MultiClassNMS, YOLOv3Head


@moduleinfo(
    name="yolov3",
    version="1.0.0",
    type="cv/object_detection",
    summary="Baidu's YOLOv3 model for object detection.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class YOLOv3(hub.Module):
    def _initialize(self):
        self.reader = reader
        self.load_label_info = load_label_info
        self.postprocess = postprocess
        self.MultiClassNMS = MultiClassNMS
        self.YOLOv3Head = YOLOv3Head

    def context(self,
                body_feats,
                yolo_head,
                image,
                trainable=True,
                var_prefix='',
                get_prediction=False):
        """
        Distill the Head Features, so as to perform transfer learning.

        Args:
            body_feats (feature maps of backbone): feature maps of backbone.
            yolo_head (<class 'YOLOv3Head' object>): yolo_head of YOLOv3
            image (Variable): image tensor.
            trainable (bool): whether to set parameters trainable.
            var_prefix (str): the prefix of variables in yolo_head and backbone.
            get_prediction (bool): whether to get prediction or not.

        Returns:
             inputs(dict): the input variables.
             outputs(dict): the output variables.
             context_prog (Program): the program to execute transfer learning.
        """
        context_prog = image.block.program
        with fluid.program_guard(context_prog):
            im_size = fluid.layers.data(
                name='im_size', shape=[2], dtype='int32')
            head_features = yolo_head._get_outputs(
                body_feats, is_train=trainable)
            inputs = {
                'image': var_prefix + image.name,
                'im_size': var_prefix + im_size.name
            }
            if get_prediction:
                bbox_out = yolo_head.get_prediction(head_features, im_size)
                outputs = {'bbox_out': [var_prefix + bbox_out.name]}
            else:
                outputs = {
                    'head_features':
                    [var_prefix + var.name for var in head_features]
                }

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            add_vars_prefix(context_prog, var_prefix)
            add_vars_prefix(fluid.default_startup_program(), var_prefix)
            inputs = {
                key: context_prog.global_block().vars[value]
                for key, value in inputs.items()
            }
            outputs = {
                key: [
                    context_prog.global_block().vars[varname]
                    for varname in value
                ]
                for key, value in outputs.items()
            }

            for param in context_prog.global_block().iter_parameters():
                param.trainable = trainable
            return inputs, outputs, context_prog
