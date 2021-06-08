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
from typing import Union

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
import paddlehub.vision.transforms as T
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
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
        self._pool2d_avg = AvgPool2D(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = Conv2D(
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
    """Bottleneck Block for ResNet50_vd."""

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
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    """Basic block for ResNet50_vd."""

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
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y


@moduleinfo(
    name="resnet50_vd_imagenet_ssld",
    type="CV/classification",
    author="paddlepaddle",
    author_email="",
    summary="resnet50_vd_imagenet_ssld is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class ResNet50_vd(nn.Layer):
    """ResNet50_vd model."""

    def __init__(self, label_list: list = None, load_checkpoint: str = None):
        super(ResNet50_vd, self).__init__()

        self.layers = 50
        if label_list is not None:
            self.labels = label_list
            class_dim = len(self.labels)
        else:
            label_list = []
            label_file = os.path.join(self.directory, 'label_list.txt')
            files = open(label_file)
            for line in files.readlines():
                line = line.strip('\n')
                label_list.append(line)
            self.labels = label_list
            class_dim = len(self.labels)
        depth = [3, 4, 6, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(num_channels=3, num_filters=32, filter_size=3, stride=2, act='relu', name="conv1_1")
        self.conv1_2 = ConvBNLayer(num_channels=32, num_filters=32, filter_size=3, stride=1, act='relu', name="conv1_2")
        self.conv1_3 = ConvBNLayer(num_channels=32, num_filters=64, filter_size=3, stride=1, act='relu', name="conv1_3")
        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []

        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):

                conv_name = "res" + str(block + 2) + chr(97 + i)
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block] if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        name=conv_name))
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1)
        self.pool2d_avg_channels = num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv), name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"))

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'resnet50_vd_ssld.pdparams')
            self.model_dict = paddle.load(checkpoint)
            self.set_dict(self.model_dict)
            print("load pretrained checkpoint success")

    def transforms(self, images: Union[str, np.ndarray]):
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ],
                               to_rgb=True)
        return transforms(images)

    def forward(self, inputs: paddle.Tensor):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        feature = self.pool2d_avg(y)
        y = paddle.reshape(feature, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y, feature
