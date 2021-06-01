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
from paddle.nn import Conv2d, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from paddle.nn.initializer import Uniform
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


class ConvBNLayer(nn.Layer):
    """Basic conv bn layer."""

    def __init__(
            self,
            num_channels: int,
            num_filters: int,
            filter_size: int,
            stride: int = 1,
            groups: int = 1,
            is_vd_mode: bool = False,
            act: str = None,
            name: str = None,
    ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs: paddle.Tensor):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    """Bottleneck Block for ResNet34_vd."""

    def __init__(self,
                 num_channels: int,
                 num_filters: int,
                 stride: int,
                 shortcut: bool = True,
                 if_first: bool = False,
                 name: str = None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels, num_filters=num_filters, filter_size=1, act='relu', name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters, num_filters=num_filters * 4, filter_size=1, act=None, name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs: paddle.Tensor):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.elementwise_add(x=short, y=conv2, act='relu')
        return y


class BasicBlock(nn.Layer):
    """Basic block for ResNet34_vd."""

    def __init__(self,
                 num_channels: int,
                 num_filters: int,
                 stride: int,
                 shortcut: bool = True,
                 if_first: bool = False,
                 name: str = None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters, num_filters=num_filters, filter_size=3, act=None, name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs: paddle.Tensor):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.elementwise_add(x=short, y=conv1, act='relu')
        return y


@moduleinfo(
    name="resnet34_vd_imagenet",
    type="CV/classification",
    author="paddlepaddle",
    author_email="",
    summary="resnet34_vd_imagenet is a classification model, "
    "this module is trained with Baidu open sourced dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class ResNet34_vd(nn.Layer):
    """ResNet34_vd model."""

    def __init__(self, class_dim: int = 1000, load_checkpoint: str = None):
        super(ResNet34_vd, self).__init__()

        self.layers = 34
        depth = [3, 4, 6, 3]
        num_channels = [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(num_channels=3, num_filters=32, filter_size=3, stride=2, act='relu', name="conv1_1")
        self.conv1_2 = ConvBNLayer(num_channels=32, num_filters=32, filter_size=3, stride=1, act='relu', name="conv1_2")
        self.conv1_3 = ConvBNLayer(num_channels=32, num_filters=64, filter_size=3, stride=1, act='relu', name="conv1_3")
        self.pool2d_max = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_list = []

        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)
                basic_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BasicBlock(
                        num_channels=num_channels[block] if i == 0 else num_filters[block],
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        name=conv_name))
                self.block_list.append(basic_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2d(1)
        self.pool2d_avg_channels = num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv), name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"))

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'resnet34_vd_imagenet.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://paddlehub.bj.bcebos.com/dygraph/image_classification/resnet34_vd_imagenet.pdparams -O '
                    + checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def forward(self, inputs: paddle.Tensor):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y
