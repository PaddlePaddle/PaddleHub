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
from paddle.regularizer import L2Decay
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


def make_divisible(v: int, divisor: int = 8, min_value: int = None):
    """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@moduleinfo(
    name="mobilenet_v3_large_imagenet_ssld",
    type="cv/classification",
    author="paddlepaddle",
    author_email="",
    summary="mobilenet_v3_large_imagenet_ssld is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.1.0",
    meta=ImageClassifierModule)
class MobileNetV3Large(nn.Layer):
    """MobileNetV3Large module."""

    def __init__(self, dropout_prob: float = 0.2, class_dim: int = 1000, load_checkpoint: str = None):
        super(MobileNetV3Large, self).__init__()

        inplanes = 16
        self.cfg = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, False, "relu", 1],
            [3, 64, 24, False, "relu", 2],
            [3, 72, 24, False, "relu", 1],
            [5, 72, 40, True, "relu", 2],
            [5, 120, 40, True, "relu", 1],
            [5, 120, 40, True, "relu", 1],
            [3, 240, 80, False, "hard_swish", 2],
            [3, 200, 80, False, "hard_swish", 1],
            [3, 184, 80, False, "hard_swish", 1],
            [3, 184, 80, False, "hard_swish", 1],
            [3, 480, 112, True, "hard_swish", 1],
            [3, 672, 112, True, "hard_swish", 1],
            [5, 672, 160, True, "hard_swish", 2],
            [5, 960, 160, True, "hard_swish", 1],
            [5, 960, 160, True, "hard_swish", 1]
        ]
        self.cls_ch_squeeze = 960
        self.cls_ch_expand = 1280

        self.conv1 = ConvBNLayer(
            in_c=3,
            out_c=make_divisible(inplanes),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hard_swish",
            name="conv1")

        self.block_list = []
        i = 0
        inplanes = make_divisible(inplanes)
        for (k, exp, c, se, nl, s) in self.cfg:
            self.block_list.append(
                ResidualUnit(
                    in_c=inplanes,
                    mid_c=make_divisible(exp),
                    out_c=make_divisible(c),
                    filter_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2)))
            self.add_sublayer(sublayer=self.block_list[-1], name="conv" + str(i + 2))
            inplanes = make_divisible(c)
            i += 1

        self.last_second_conv = ConvBNLayer(
            in_c=inplanes,
            out_c=make_divisible(self.cls_ch_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hard_swish",
            name="conv_last")

        self.pool = AdaptiveAvgPool2d(1)

        self.last_conv = Conv2d(
            in_channels=make_divisible(self.cls_ch_squeeze),
            out_channels=self.cls_ch_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="last_1x1_conv_weights"),
            bias_attr=False)

        self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")

        self.out = Linear(
            self.cls_ch_expand, class_dim, weight_attr=ParamAttr("fc_weights"), bias_attr=ParamAttr(name="fc_offset"))

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'mobilenet_v3_large_ssld.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://paddlehub.bj.bcebos.com/dygraph/image_classification/mobilenet_v3_large_ssld.pdparams -O '
                    + checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def forward(self, inputs: paddle.Tensor):
        x = self.conv1(inputs)
        for block in self.block_list:
            x = block(x)

        x = self.last_second_conv(x)
        x = self.pool(x)

        x = self.last_conv(x)
        x = F.hard_swish(x)
        x = self.dropout(x)
        x = paddle.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.out(x)
        return x


class ConvBNLayer(nn.Layer):
    """Basic conv bn layer."""

    def __init__(self,
                 in_c: int,
                 out_c: int,
                 filter_size: int,
                 stride: int,
                 padding: int,
                 num_groups: int = 1,
                 if_act: bool = True,
                 act: str = None,
                 name: str = ""):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(name=name + "_bn_scale", regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(name=name + "_bn_offset", regularizer=L2Decay(0.0)),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, x: paddle.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hard_swish":
                x = F.hard_swish(x)
            else:
                print("The activation function is selected incorrectly.")
                exit()
        return x


class ResidualUnit(nn.Layer):
    """Residual unit for MobileNetV3."""

    def __init__(self,
                 in_c: int,
                 mid_c: int,
                 out_c: int,
                 filter_size: int,
                 stride: int,
                 use_se: int,
                 act: str = None,
                 name: str = ''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c, out_c=mid_c, filter_size=1, stride=1, padding=0, if_act=True, act=act, name=name + "_expand")
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_c=mid_c, out_c=out_c, filter_size=1, stride=1, padding=0, if_act=False, act=None, name=name + "_linear")

    def forward(self, inputs: paddle.Tensor):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.elementwise_add(inputs, x)
        return x


class SEModule(nn.Layer):
    """Basic model for ResidualUnit."""

    def __init__(self, channel: int, reduction: int = 4, name: str = ""):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs: paddle.Tensor):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hard_sigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs, axis=0)
