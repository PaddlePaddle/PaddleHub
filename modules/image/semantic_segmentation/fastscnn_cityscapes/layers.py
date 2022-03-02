# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    """Basic conv bn relu layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Layer):
    """Basic conv bn layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):
    """Basic conv bn pool layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = F.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class SeparableConvBNReLU(nn.Layer):
    """Basic separable conv bn relu layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = ConvBNReLU(in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    """Basic depthwise conv bn relu layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self, in_channels: int, inter_channels: int, out_channels: int, dropout_prob: float = 0.1):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(in_channels=inter_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class Activation(nn.Layer):
    """
    The wrapper of activations.
    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.
    Returns:
        A callable object of Activation.
    Raises:
        KeyError: When parameter `act` is not in the optional range.
    Examples:
        from paddleseg.models.common.activation import Activation
        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>
        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>
        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """

    def __init__(self, act: str = None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.layer.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("nn.layer.activation.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(act, act_dict.keys()))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self._act is not None:
            return self.act_func(x)
        else:
            return x


class PPModule(nn.Layer):
    """
    Pyramid pooling module originally in PSPNet.

    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, in_channels: int, out_channels: int, bin_sizes: Tuple, dim_reduction: bool, align_corners: bool):
        super().__init__()

        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([self._make_stage(in_channels, inter_channels, size) for size in bin_sizes])

        self.conv_bn_relu2 = ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels: int, out_channels: int, size: int):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.

        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            out_channels (int): The number of output channels to pyramid pooling module.
            size (int): The out size of the pooled layer.

        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """

        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        return nn.Sequential(prior, conv)

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(x, paddle.shape(input)[2:], mode='bilinear', align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out
