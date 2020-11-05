# coding=utf-8
from __future__ import absolute_import

import ast
import argparse
import os
from functools import partial

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving
from paddlehub.common.paddle_helper import add_vars_prefix

from yolov3_darknet53_venus.darknet import DarkNet
from yolov3_darknet53_venus.processor import load_label_info, postprocess, base64_to_cv2
from yolov3_darknet53_venus.data_feed import reader
from yolov3_darknet53_venus.yolo_head import MultiClassNMS, YOLOv3Head


@moduleinfo(
    name="yolov3_darknet53_venus",
    version="1.0.0",
    type="CV/object_detection",
    summary="Baidu's YOLOv3 model for object detection, with backbone DarkNet53, trained with Baidu self-built dataset.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class YOLOv3DarkNet53Venus(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "yolov3_darknet53_model")

    def context(self, trainable=True, pretrained=True, get_prediction=False):
        """
        Distill the Head Features, so as to perform transfer learning.

        Args:
            trainable (bool): whether to set parameters trainable.
            pretrained (bool): whether to load default pretrained model.
            get_prediction (bool): whether to get prediction.

        Returns:
             inputs(dict): the input variables.
             outputs(dict): the output variables.
             context_prog (Program): the program to execute transfer learning.
        """
        context_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(context_prog, startup_program):
            with fluid.unique_name.guard():
                # image
                image = fluid.layers.data(name='image', shape=[3, 608, 608], dtype='float32')
                # backbone
                backbone = DarkNet(norm_type='bn', norm_decay=0., depth=53)
                # body_feats
                body_feats = backbone(image)
                # im_size
                im_size = fluid.layers.data(name='im_size', shape=[2], dtype='int32')
                # yolo_head
                yolo_head = YOLOv3Head(num_classes=708)
                # head_features
                head_features, body_features = yolo_head._get_outputs(body_feats, is_train=trainable)

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())

                # var_prefix
                var_prefix = '@HUB_{}@'.format(self.name)
                # name of inputs
                inputs = {'image': var_prefix + image.name, 'im_size': var_prefix + im_size.name}
                # name of outputs
                if get_prediction:
                    bbox_out = yolo_head.get_prediction(head_features, im_size)
                    outputs = {'bbox_out': [var_prefix + bbox_out.name]}
                else:
                    outputs = {
                        'head_features': [var_prefix + var.name for var in head_features],
                        'body_features': [var_prefix + var.name for var in body_features]
                    }
                # add_vars_prefix
                add_vars_prefix(context_prog, var_prefix)
                add_vars_prefix(fluid.default_startup_program(), var_prefix)
                # inputs
                inputs = {key: context_prog.global_block().vars[value] for key, value in inputs.items()}
                # outputs
                outputs = {
                    key: [context_prog.global_block().vars[varname] for varname in value]
                    for key, value in outputs.items()
                }
                # trainable
                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(os.path.join(self.default_pretrained_model_path, var.name))

                    fluid.io.load_vars(exe, self.default_pretrained_model_path, predicate=_if_exist)
                else:
                    exe.run(startup_program)

                return inputs, outputs, context_prog
