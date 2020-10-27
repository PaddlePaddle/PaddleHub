# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2d, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from paddle.nn.initializer import MSRA
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


class ConvBNLayer(nn.Layer):
    """Basic conv bn layer."""

    def __init__(self,
                 num_channels: int,
                 filter_size: int,
                 num_filters: int,
                 stride: int,
                 padding: int,
                 channels: int = None,
                 num_groups: int = 1,
                 act: str = 'relu',
                 name: str = None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=MSRA(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name + "_bn_scale"),
            bias_attr=ParamAttr(name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs: paddle.Tensor):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DepthwiseSeparable(nn.Layer):
    """Depthwise and pointwise conv layer."""

    def __init__(self,
                 num_channels: int,
                 num_filters1: int,
                 num_filters2: int,
                 num_groups: int,
                 stride: int,
                 scale: float,
                 name: str = None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            name=name + "_dw")

        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep")

    def forward(self, inputs: paddle.Tensor):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


@moduleinfo(
    name="mobilenet_v1_imagenet",
    type="cv/classification",
    author="paddlepaddle",
    author_email="",
    summary="mobilenet_v1_imagenet is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class MobileNetV1(nn.Layer):
    """MobileNetV1"""

    def __init__(self, class_dim: int = 1000, load_checkpoint: str = None):
        super(MobileNetV1, self).__init__()
        self.block_list = []

        self.conv1 = ConvBNLayer(
            num_channels=3, filter_size=3, channels=3, num_filters=int(32), stride=2, padding=1, name="conv1")

        conv2_1 = self.add_sublayer(
            "conv2_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(32),
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=1,
                name="conv2_1"))
        self.block_list.append(conv2_1)

        conv2_2 = self.add_sublayer(
            "conv2_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(64),
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=2,
                scale=1,
                name="conv2_2"))
        self.block_list.append(conv2_2)

        conv3_1 = self.add_sublayer(
            "conv3_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(128),
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=1,
                name="conv3_1"))
        self.block_list.append(conv3_1)

        conv3_2 = self.add_sublayer(
            "conv3_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(128),
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=2,
                scale=1,
                name="conv3_2"))
        self.block_list.append(conv3_2)

        conv4_1 = self.add_sublayer(
            "conv4_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(256),
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=1,
                name="conv4_1"))
        self.block_list.append(conv4_1)

        conv4_2 = self.add_sublayer(
            "conv4_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(256),
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=2,
                scale=1,
                name="conv4_2"))
        self.block_list.append(conv4_2)

        for i in range(5):
            conv5 = self.add_sublayer(
                "conv5_" + str(i + 1),
                sublayer=DepthwiseSeparable(
                    num_channels=int(512),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    scale=1,
                    name="conv5_" + str(i + 1)))
            self.block_list.append(conv5)

        conv5_6 = self.add_sublayer(
            "conv5_6",
            sublayer=DepthwiseSeparable(
                num_channels=int(512),
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=2,
                scale=1,
                name="conv5_6"))
        self.block_list.append(conv5_6)

        conv6 = self.add_sublayer(
            "conv6",
            sublayer=DepthwiseSeparable(
                num_channels=int(1024),
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                scale=1,
                name="conv6"))
        self.block_list.append(conv6)

        self.pool2d_avg = AdaptiveAvgPool2d(1)

        self.out = Linear(
            int(1024),
            class_dim,
            weight_attr=ParamAttr(initializer=MSRA(), name="fc7_weights"),
            bias_attr=ParamAttr(name="fc7_offset"))

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'mobilenet_v1_imagenet.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://paddlehub.bj.bcebos.com/dygraph/image_classification/mobilenet_v1_imagenet.pdparams -O '
                    + checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def forward(self, inputs: paddle.Tensor):
        y = self.conv1(inputs)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, 1024])
        y = self.out(y)
        return y
