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
import sys
import math

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2d, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from paddle.nn.initializer import Uniform
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


class ConvBNLayer(nn.Layer):
    """Basic conv bn layer."""

    def __init__(self,
                 num_channels: int,
                 num_filters: int,
                 filter_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 act: str = None,
                 name: str = None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = "bn_" + name
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs: paddle.Tensor):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class SeparableConv(nn.Layer):
    """Basic separable conv layer, it contains pointwise conv and depthwise conv."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1, name: str = None):
        super(SeparableConv, self).__init__()

        self._pointwise_conv = ConvBNLayer(input_channels, output_channels, 1, name=name + "_sep")
        self._depthwise_conv = ConvBNLayer(
            output_channels, output_channels, 3, stride=stride, groups=output_channels, name=name + "_dw")

    def forward(self, inputs: paddle.Tensor):
        x = self._pointwise_conv(inputs)
        x = self._depthwise_conv(x)
        return x


class EntryFlowBottleneckBlock(nn.Layer):
    """Basic entry flow bottleneck block for Xception."""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 stride: int = 2,
                 name: str = None,
                 relu_first: bool = False):
        super(EntryFlowBottleneckBlock, self).__init__()
        self.relu_first = relu_first

        self._short = Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            weight_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False)
        self._conv1 = SeparableConv(input_channels, output_channels, stride=1, name=name + "_branch2a_weights")
        self._conv2 = SeparableConv(output_channels, output_channels, stride=1, name=name + "_branch2b_weights")
        self._pool = MaxPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, inputs: paddle.Tensor):
        conv0 = inputs
        short = self._short(inputs)
        if self.relu_first:
            conv0 = F.relu(conv0)
        conv1 = self._conv1(conv0)
        conv2 = F.relu(conv1)
        conv2 = self._conv2(conv2)
        pool = self._pool(conv2)
        return paddle.elementwise_add(x=short, y=pool)


class EntryFlow(nn.Layer):
    """Entry flow for Xception."""

    def __init__(self, block_num: int = 3):
        super(EntryFlow, self).__init__()

        name = "entry_flow"
        self.block_num = block_num
        self._conv1 = ConvBNLayer(3, 32, 3, stride=2, act="relu", name=name + "_conv1")
        self._conv2 = ConvBNLayer(32, 64, 3, act="relu", name=name + "_conv2")
        if block_num == 3:
            self._conv_0 = EntryFlowBottleneckBlock(64, 128, stride=2, name=name + "_0", relu_first=False)
            self._conv_1 = EntryFlowBottleneckBlock(128, 256, stride=2, name=name + "_1", relu_first=True)
            self._conv_2 = EntryFlowBottleneckBlock(256, 728, stride=2, name=name + "_2", relu_first=True)
        elif block_num == 5:
            self._conv_0 = EntryFlowBottleneckBlock(64, 128, stride=2, name=name + "_0", relu_first=False)
            self._conv_1 = EntryFlowBottleneckBlock(128, 256, stride=1, name=name + "_1", relu_first=True)
            self._conv_2 = EntryFlowBottleneckBlock(256, 256, stride=2, name=name + "_2", relu_first=True)
            self._conv_3 = EntryFlowBottleneckBlock(256, 728, stride=1, name=name + "_3", relu_first=True)
            self._conv_4 = EntryFlowBottleneckBlock(728, 728, stride=2, name=name + "_4", relu_first=True)
        else:
            sys.exit(-1)

    def forward(self, inputs: paddle.Tensor):
        x = self._conv1(inputs)
        x = self._conv2(x)

        if self.block_num == 3:
            x = self._conv_0(x)
            x = self._conv_1(x)
            x = self._conv_2(x)
        elif self.block_num == 5:
            x = self._conv_0(x)
            x = self._conv_1(x)
            x = self._conv_2(x)
            x = self._conv_3(x)
            x = self._conv_4(x)
        return x


class MiddleFlowBottleneckBlock(nn.Layer):
    """Basic middle flow bottleneck block for Xception."""

    def __init__(self, input_channels: int, output_channels: int, name: str):
        super(MiddleFlowBottleneckBlock, self).__init__()

        self._conv_0 = SeparableConv(input_channels, output_channels, stride=1, name=name + "_branch2a_weights")
        self._conv_1 = SeparableConv(output_channels, output_channels, stride=1, name=name + "_branch2b_weights")
        self._conv_2 = SeparableConv(output_channels, output_channels, stride=1, name=name + "_branch2c_weights")

    def forward(self, inputs: paddle.Tensor):
        conv0 = F.relu(inputs)
        conv0 = self._conv_0(conv0)
        conv1 = F.relu(conv0)
        conv1 = self._conv_1(conv1)
        conv2 = F.relu(conv1)
        conv2 = self._conv_2(conv2)
        return paddle.elementwise_add(x=inputs, y=conv2)


class MiddleFlow(nn.Layer):
    """Middle flow for Xception."""

    def __init__(self, block_num: int = 8):
        super(MiddleFlow, self).__init__()

        self.block_num = block_num
        self._conv_0 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_0")
        self._conv_1 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_1")
        self._conv_2 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_2")
        self._conv_3 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_3")
        self._conv_4 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_4")
        self._conv_5 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_5")
        self._conv_6 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_6")
        self._conv_7 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_7")
        if block_num == 16:
            self._conv_8 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_8")
            self._conv_9 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_9")
            self._conv_10 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_10")
            self._conv_11 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_11")
            self._conv_12 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_12")
            self._conv_13 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_13")
            self._conv_14 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_14")
            self._conv_15 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_15")

    def forward(self, inputs: paddle.Tensor):
        x = self._conv_0(inputs)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._conv_3(x)
        x = self._conv_4(x)
        x = self._conv_5(x)
        x = self._conv_6(x)
        x = self._conv_7(x)
        if self.block_num == 16:
            x = self._conv_8(x)
            x = self._conv_9(x)
            x = self._conv_10(x)
            x = self._conv_11(x)
            x = self._conv_12(x)
            x = self._conv_13(x)
            x = self._conv_14(x)
            x = self._conv_15(x)
        return x


class ExitFlowBottleneckBlock(nn.Layer):
    """Basic exit flow bottleneck block for Xception."""

    def __init__(self, input_channels: int, output_channels1: int, output_channels2: int, name: str):
        super(ExitFlowBottleneckBlock, self).__init__()

        self._short = Conv2d(
            in_channels=input_channels,
            out_channels=output_channels2,
            kernel_size=1,
            stride=2,
            padding=0,
            weight_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False)
        self._conv_1 = SeparableConv(input_channels, output_channels1, stride=1, name=name + "_branch2a_weights")
        self._conv_2 = SeparableConv(output_channels1, output_channels2, stride=1, name=name + "_branch2b_weights")
        self._pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: paddle.Tensor):
        short = self._short(inputs)
        conv0 = F.relu(inputs)
        conv1 = self._conv_1(conv0)
        conv2 = F.relu(conv1)
        conv2 = self._conv_2(conv2)
        pool = self._pool(conv2)
        return paddle.elementwise_add(x=short, y=pool)


class ExitFlow(nn.Layer):
    """Exit flow for Xception."""

    def __init__(self, class_dim: int):
        super(ExitFlow, self).__init__()

        name = "exit_flow"

        self._conv_0 = ExitFlowBottleneckBlock(728, 728, 1024, name=name + "_1")
        self._conv_1 = SeparableConv(1024, 1536, stride=1, name=name + "_2")
        self._conv_2 = SeparableConv(1536, 2048, stride=1, name=name + "_3")
        self._pool = AdaptiveAvgPool2d(1)
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self._out = Linear(
            2048,
            class_dim,
            weight_attr=ParamAttr(name="fc_weights", initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, inputs: paddle.Tensor):
        conv0 = self._conv_0(inputs)
        conv1 = self._conv_1(conv0)
        conv1 = F.relu(conv1)
        conv2 = self._conv_2(conv1)
        conv2 = F.relu(conv2)
        pool = self._pool(conv2)
        pool = paddle.reshape(pool, [0, -1])
        out = self._out(pool)
        return out


@moduleinfo(
    name="xception41_imagenet",
    type="CV/classification",
    author="paddlepaddle",
    author_email="",
    summary="Xception41_imagenet is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class Xception41(nn.Layer):
    """Xception41 model."""

    def __init__(self, class_dim: int = 1000, load_checkpoint: str = None):
        super(Xception41, self).__init__()
        self.entry_flow_block_num = 3
        self.middle_flow_block_num = 8
        self._entry_flow = EntryFlow(self.entry_flow_block_num)
        self._middle_flow = MiddleFlow(self.middle_flow_block_num)
        self._exit_flow = ExitFlow(class_dim)

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'xception41_imagenet.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://paddlehub.bj.bcebos.com/dygraph/image_classification/xception41_imagenet.pdparams -O'
                    + checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def forward(self, inputs: paddle.Tensor):
        x = self._entry_flow(inputs)
        x = self._middle_flow(x)
        x = self._exit_flow(x)
        return x
