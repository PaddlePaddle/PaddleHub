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
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer import activation
from paddle.nn import Conv2D, AvgPool2D


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu':
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNLayer(nn.Layer):
    """Basic conv bn relu layer."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 is_vd_mode: bool = False,
                 act: str = None,
                 name: str = None):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = AvgPool2D(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if dilation == 1 else 0,
            dilation=dilation,
            groups=groups,
            bias_attr=False)

        self._batch_norm = SyncBatchNorm(out_channels)
        self._act_op = Activation(act=act)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y


class BottleneckBlock(nn.Layer):
    """Residual bottleneck block"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 shortcut: bool = True,
                 if_first: bool = False,
                 dilation: int = 1,
                 name: str = None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, act='relu', name=name + "_branch2a")

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, act=None, name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        y = self.conv0(inputs)
        if self.dilation > 1:
            padding = self.dilation
            y = F.pad(y, [padding, padding, padding, padding])

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class SeparableConvBNReLU(nn.Layer):
    """Depthwise Separable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs: dict):
        super(SeparableConvBNReLU, self).__init__()
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


class ConvBN(nn.Layer):
    """Basic conv bn layer"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs: dict):
        super(ConvBN, self).__init__()
        self._conv = Conv2D(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvBNReLU(nn.Layer):
    """Basic conv bn relu layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'same', **kwargs: dict):
        super(ConvBNReLU, self).__init__()

        self._conv = Conv2D(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
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


class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios: Tuple[int],
                 in_channels: int,
                 out_channels: int,
                 align_corners: bool,
                 use_sep_conv: bool = False,
                 image_pooling: bool = False):
        super().__init__()

        self.align_corners = align_corners
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(output_size=(1, 1)),
                ConvBNReLU(in_channels, out_channels, kernel_size=1, bias_attr=False))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = ConvBNReLU(in_channels=out_channels * out_size, out_channels=out_channels, kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        outputs = []
        for block in self.aspp_blocks:
            y = block(x)
            y = F.interpolate(y, x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(img_avg, x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=1)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x
