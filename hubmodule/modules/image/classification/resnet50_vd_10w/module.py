# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import argparse
import os

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving
from paddlehub.common.paddle_helper import add_vars_prefix

from resnet50_vd_10w.processor import postprocess, base64_to_cv2
from resnet50_vd_10w.data_feed import reader
from resnet50_vd_10w.resnet_vd import ResNet50_vd


@moduleinfo(
    name="resnet50_vd_10w",
    type="CV/image_classification",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    summary=
    "ResNet50vd is a image classfication model, this module is trained with Baidu's self-built dataset with 100,000 categories.",
    version="1.0.0")
class ResNet50vd(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "model")

    def get_expected_image_width(self):
        return 224

    def get_expected_image_height(self):
        return 224

    def get_pretrained_images_mean(self):
        im_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3)
        return im_mean

    def get_pretrained_images_std(self):
        im_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3)
        return im_std

    def context(self, trainable=True, pretrained=True):
        """context for transfer learning.

        Args:
            trainable (bool): Set parameters in program to be trainable.
            pretrained (bool) : Whether to load pretrained model.

        Returns:
            inputs (dict): key is 'image', corresponding vaule is image tensor.
            outputs (dict): key is 'feature_map', corresponding value is the result of the layer before the fully connected layer.
            context_prog (fluid.Program): program for transfer learning.
        """
        context_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(context_prog, startup_prog):
            with fluid.unique_name.guard():
                image = fluid.layers.data(
                    name="image", shape=[3, 224, 224], dtype="float32")
                resnet_vd = ResNet50_vd()
                feature_map = resnet_vd.net(input=image)

                name_prefix = '@HUB_{}@'.format(self.name)
                inputs = {'image': name_prefix + image.name}
                outputs = {'feature_map': name_prefix + feature_map.name}
                add_vars_prefix(context_prog, name_prefix)
                add_vars_prefix(startup_prog, name_prefix)
                global_vars = context_prog.global_block().vars
                inputs = {
                    key: global_vars[value]
                    for key, value in inputs.items()
                }
                outputs = {
                    key: global_vars[value]
                    for key, value in outputs.items()
                }

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        b = os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))
                        return b

                    fluid.io.load_vars(
                        exe,
                        self.default_pretrained_model_path,
                        context_prog,
                        predicate=_if_exist)
                else:
                    exe.run(startup_prog)
                # trainable
                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable
        return inputs, outputs, context_prog

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.default_pretrained_model_path, executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
            main_program=program,
            executor=exe,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            model_filename=model_filename,
            params_filename=params_filename)
