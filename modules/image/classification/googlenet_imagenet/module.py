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

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2d, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from paddle.nn.initializer import Uniform
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


def xavier(channels: int, filter_size: int, name: str):
    """Initialize the weights by uniform distribution."""
    stdv = (3.0 / (filter_size**2 * channels))**0.5
    param_attr = ParamAttr(initializer=Uniform(-stdv, stdv), name=name + "_weights")
    return param_attr


class ConvLayer(nn.Layer):
    """Basic conv2d layer."""

    def __init__(self,
                 num_channels: int,
                 num_filters: int,
                 filter_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 name: str = None):
        super(ConvLayer, self).__init__()

        self._conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)

    def forward(self, inputs: paddle.Tensor):
        y = self._conv(inputs)
        return y


class Inception(nn.Layer):
    """Inception block."""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 filter1: int,
                 filter3R: int,
                 filter3: int,
                 filter5R: int,
                 filter5: int,
                 proj: int,
                 name: str = None):
        super(Inception, self).__init__()

        self._conv1 = ConvLayer(input_channels, filter1, 1, name="inception_" + name + "_1x1")
        self._conv3r = ConvLayer(input_channels, filter3R, 1, name="inception_" + name + "_3x3_reduce")
        self._conv3 = ConvLayer(filter3R, filter3, 3, name="inception_" + name + "_3x3")
        self._conv5r = ConvLayer(input_channels, filter5R, 1, name="inception_" + name + "_5x5_reduce")
        self._conv5 = ConvLayer(filter5R, filter5, 5, name="inception_" + name + "_5x5")
        self._pool = MaxPool2d(kernel_size=3, stride=1, padding=1)

        self._convprj = ConvLayer(input_channels, proj, 1, name="inception_" + name + "_3x3_proj")

    def forward(self, inputs: paddle.Tensor):
        conv1 = self._conv1(inputs)

        conv3r = self._conv3r(inputs)
        conv3 = self._conv3(conv3r)

        conv5r = self._conv5r(inputs)
        conv5 = self._conv5(conv5r)

        pool = self._pool(inputs)
        convprj = self._convprj(pool)

        cat = paddle.concat([conv1, conv3, conv5, convprj], axis=1)
        cat = F.relu(cat)
        return cat


@moduleinfo(
    name="googlenet_imagenet",
    type="CV/classification",
    author="paddlepaddle",
    author_email="",
    summary="GoogleNet_imagenet is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class GoogleNet(nn.Layer):
    """GoogleNet model"""

    def __init__(self, class_dim: int = 1000, load_checkpoint: str = None):
        super(GoogleNet, self).__init__()
        self._conv = ConvLayer(3, 64, 7, 2, name="conv1")
        self._pool = MaxPool2d(kernel_size=3, stride=2)
        self._conv_1 = ConvLayer(64, 64, 1, name="conv2_1x1")
        self._conv_2 = ConvLayer(64, 192, 3, name="conv2_3x3")

        self._ince3a = Inception(192, 192, 64, 96, 128, 16, 32, 32, name="ince3a")
        self._ince3b = Inception(256, 256, 128, 128, 192, 32, 96, 64, name="ince3b")

        self._ince4a = Inception(480, 480, 192, 96, 208, 16, 48, 64, name="ince4a")
        self._ince4b = Inception(512, 512, 160, 112, 224, 24, 64, 64, name="ince4b")
        self._ince4c = Inception(512, 512, 128, 128, 256, 24, 64, 64, name="ince4c")
        self._ince4d = Inception(512, 512, 112, 144, 288, 32, 64, 64, name="ince4d")
        self._ince4e = Inception(528, 528, 256, 160, 320, 32, 128, 128, name="ince4e")

        self._ince5a = Inception(832, 832, 256, 160, 320, 32, 128, 128, name="ince5a")
        self._ince5b = Inception(832, 832, 384, 192, 384, 48, 128, 128, name="ince5b")

        self._pool_5 = AvgPool2d(kernel_size=7, stride=7)

        self._drop = Dropout(p=0.4, mode="downscale_in_infer")
        self._fc_out = Linear(
            1024, class_dim, weight_attr=xavier(1024, 1, "out"), bias_attr=ParamAttr(name="out_offset"))

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'googlenet_imagenet.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://paddlehub.bj.bcebos.com/dygraph/image_classification/googlenet_imagenet.pdparams -O' +
                    checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def forward(self, inputs: paddle.Tensor):
        x = self._conv(inputs)
        x = self._pool(x)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._pool(x)

        x = self._ince3a(x)
        x = self._ince3b(x)
        x = self._pool(x)

        ince4a = self._ince4a(x)
        x = self._ince4b(ince4a)
        x = self._ince4c(x)
        ince4d = self._ince4d(x)
        x = self._ince4e(ince4d)
        x = self._pool(x)

        x = self._ince5a(x)
        ince5b = self._ince5b(x)

        x = self._pool_5(ince5b)
        x = self._drop(x)
        x = paddle.squeeze(x, axis=[2, 3])
        out = self._fc_out(x)
        out = F.softmax(out)

        return out
