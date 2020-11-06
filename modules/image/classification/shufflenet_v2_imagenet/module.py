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
from paddle.nn.initializer import MSRA
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


def channel_shuffle(x: paddle.Tensor, groups: int):
    """Shuffle input channels."""
    batchsize, num_channels, height, width = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = num_channels // groups

    # reshape
    x = paddle.reshape(x=x, shape=[batchsize, groups, channels_per_group, height, width])

    x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])
    # flatten
    x = paddle.reshape(x=x, shape=[batchsize, num_channels, height, width])
    return x


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
                 if_act: bool = True,
                 act: str = 'relu',
                 name: str = None):
        super(ConvBNLayer, self).__init__()
        self._if_act = if_act
        assert act in ['relu', 'swish'], \
            "supported act are {} but your act is {}".format(
                ['relu', 'swish'], act)
        self._act = act
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
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs: paddle.Tensor, if_act: bool = True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._if_act:
            y = F.relu(y) if self._act == 'relu' else F.swish(y)
        return y


class InvertedResidualUnit(nn.Layer):
    """Inverted Residual unit."""

    def __init__(self,
                 num_channels: int,
                 num_filters: int,
                 stride: int,
                 benchmodel: int,
                 act: str = 'relu',
                 name: str = None):
        super(InvertedResidualUnit, self).__init__()
        assert stride in [1, 2], \
            "supported stride are {} but your stride is {}".format([1, 2], stride)
        self.benchmodel = benchmodel
        oup_inc = num_filters // 2
        inp = num_channels
        if benchmodel == 1:
            self._conv_pw = ConvBNLayer(
                num_channels=num_channels // 2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv1')
            self._conv_dw = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                act=act,
                name='stage_' + name + '_conv2')
            self._conv_linear = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv3')
        else:
            # branch1
            self._conv_dw_1 = ConvBNLayer(
                num_channels=num_channels,
                num_filters=inp,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=inp,
                if_act=False,
                act=act,
                name='stage_' + name + '_conv4')
            self._conv_linear_1 = ConvBNLayer(
                num_channels=inp,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv5')
            # branch2
            self._conv_pw_2 = ConvBNLayer(
                num_channels=num_channels,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv1')
            self._conv_dw_2 = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                act=act,
                name='stage_' + name + '_conv2')
            self._conv_linear_2 = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv3')

    def forward(self, inputs: paddle.Tensor):
        if self.benchmodel == 1:
            x1, x2 = paddle.split(inputs, num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2], axis=1)
            x2 = self._conv_pw(x2)
            x2 = self._conv_dw(x2)
            x2 = self._conv_linear(x2)
            out = paddle.concat([x1, x2], axis=1)
        else:
            x1 = self._conv_dw_1(inputs)
            x1 = self._conv_linear_1(x1)

            x2 = self._conv_pw_2(inputs)
            x2 = self._conv_dw_2(x2)
            x2 = self._conv_linear_2(x2)
            out = paddle.concat([x1, x2], axis=1)

        return channel_shuffle(out, 2)


@moduleinfo(
    name="shufflenet_v2_imagenet",
    type="cv/classification",
    author="paddlepaddle",
    author_email="",
    summary="shufflenet_v2_imagenet is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class ShuffleNet(nn.Layer):
    """ShuffleNet model."""

    def __init__(self, class_dim: int = 1000, load_checkpoint: str = None):
        super(ShuffleNet, self).__init__()
        self.scale = 1
        self.class_dim = class_dim
        stage_repeats = [4, 8, 4]
        stage_out_channels = [-1, 24, 116, 232, 464, 1024]

        # 1. conv1
        self._conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=stage_out_channels[1],
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name='stage1_conv')
        self._max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. bottleneck sequences
        self._block_list = []
        i = 1
        in_c = int(32)
        for idxstage in range(len(stage_repeats)):
            numrepeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    block = self.add_sublayer(
                        str(idxstage + 2) + '_' + str(i + 1),
                        InvertedResidualUnit(
                            num_channels=stage_out_channels[idxstage + 1],
                            num_filters=output_channel,
                            stride=2,
                            benchmodel=2,
                            act='relu',
                            name=str(idxstage + 2) + '_' + str(i + 1)))
                    self._block_list.append(block)
                else:
                    block = self.add_sublayer(
                        str(idxstage + 2) + '_' + str(i + 1),
                        InvertedResidualUnit(
                            num_channels=output_channel,
                            num_filters=output_channel,
                            stride=1,
                            benchmodel=1,
                            act='relu',
                            name=str(idxstage + 2) + '_' + str(i + 1)))
                    self._block_list.append(block)

        # 3. last_conv
        self._last_conv = ConvBNLayer(
            num_channels=stage_out_channels[-2],
            num_filters=stage_out_channels[-1],
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act='relu',
            name='conv5')

        # 4. pool
        self._pool2d_avg = AdaptiveAvgPool2d(1)
        self._out_c = stage_out_channels[-1]
        # 5. fc
        self._fc = Linear(
            stage_out_channels[-1],
            class_dim,
            weight_attr=ParamAttr(name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))
        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'shufflenet_v2_imagenet.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://paddlehub.bj.bcebos.com/dygraph/image_classification/shufflenet_v2_imagenet.pdparams -O '
                    + checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def forward(self, inputs: paddle.Tensor):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        for inv in self._block_list:
            y = inv(y)
        y = self._last_conv(y)
        y = self._pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self._out_c])
        y = self._fc(y)
        return y
