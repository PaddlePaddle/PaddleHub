# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import math

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from paddleseg import utils
from paddleseg.cvlibs import manager


__all__ = ["MobileNetV2"]


class ConvBNLayer(nn.Layer):
    """Basic conv bn relu layer."""
    def __init__(self,
                 num_channels: int,
                 filter_size: int,
                 num_filters: int,
                 stride: int,
                 padding: int,
                 num_groups: int=1,
                 name: str = None,
                 use_cudnn: bool = True):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs: paddle.Tensor, if_act: bool = True) -> paddle.Tensor:
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = F.relu6(y)
        return y


class InvertedResidualUnit(nn.Layer):
    """Inverted residual block"""
    def __init__(self, num_channels: int, num_in_filter: int, num_filters: int, stride: int,
                 filter_size: int, padding: int, expansion_factor: int, name: str):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=name + "_expand")

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False,
            name=name + "_dwise")

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=name + "_linear")

    def forward(self, inputs: paddle.Tensor, ifshortcut: bool) -> paddle.Tensor:
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = paddle.add(inputs, y)
        return y


class InvresiBlocks(nn.Layer):
    def __init__(self, in_c: int, t: int, c: int, n: int, s: int, name: str):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + "_1")

        self._block_list = []
        for i in range(1, n):
            block = self.add_sublayer(
                name + "_" + str(i + 1),
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t,
                    name=name + "_" + str(i + 1)))
            self._block_list.append(block)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        y = self._first_block(inputs, ifshortcut=False)
        for block in self._block_list:
            y = block(y, ifshortcut=True)
        return y


class MobileNet(nn.Layer):
    """Networj of MobileNet"""
    def __init__(self,
                 input_channels: int = 3,
                 scale: float = 1.0,
                 pretrained: str = None,
                 prefix_name: str = ""):
        super(MobileNet, self).__init__()
        self.scale = scale

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.conv1 = ConvBNLayer(
            num_channels=input_channels,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            name=prefix_name + "conv1_1")

        self.block_list = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            block = self.add_sublayer(
                prefix_name + "conv" + str(i),
                sublayer=InvresiBlocks(
                    in_c=in_c,
                    t=t,
                    c=int(c * scale),
                    n=n,
                    s=s,
                    name=prefix_name + "conv" + str(i)))
            self.block_list.append(block)
            in_c = int(c * scale)

        self.out_c = int(1280 * scale) if scale > 1.0 else 1280
        self.conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self.out_c,
            filter_size=1,
            stride=1,
            padding=0,
            name=prefix_name + "conv9")

        self.feat_channels = [int(i * scale) for i in [16, 24, 32, 96, 1280]]
        self.pretrained = pretrained

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        feat_list = []
        y = self.conv1(inputs, if_act=True)

        block_index = 0
        for block in self.block_list:
            y = block(y)
            if block_index in [0, 1, 2, 4]:
                feat_list.append(y)
            block_index += 1
        y = self.conv9(y, if_act=True)
        feat_list.append(y)
        return feat_list


def MobileNetV2(**kwargs):
    model = MobileNet(scale=1.0, **kwargs)
    return model
