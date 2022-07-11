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


class SeparableConvBNReLU(nn.Layer):
    """Depthwise Separable Convolution."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: str = 'same',
                 **kwargs: dict):
        super(SeparableConvBNReLU, self).__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = ConvBNReLU(
            in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class ConvBN(nn.Layer):
    """Basic conv bn layer"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: str = 'same',
                 **kwargs: dict):
        super(ConvBN, self).__init__()
        self._conv = Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvBNReLU(nn.Layer):
    """Basic conv bn relu layer."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: str = 'same',
                 **kwargs: dict):
        super(ConvBNReLU, self).__init__()

        self._conv = Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
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
        upper_act_names = activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("activation.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

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
                 aspp_ratios: tuple,
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

        self.conv_bn_relu = ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        outputs = []
        for block in self.aspp_blocks:
            y = block(x)
            y = F.interpolate(
                y,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=1)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

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

    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 dropout_prob: float = 0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class Add(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: paddle.Tensor, y: paddle.Tensor, name: str = None):
        return paddle.add(x, y, name)

class AttentionBlock(nn.Layer):
    """General self-attention block/non-local block.

    The original article refers to refer to https://arxiv.org/abs/1706.03762.
    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_out_num_convs (int): Number of convs for value projection.
        key_query_norm (bool): Whether to use BN for key/query projection.
        value_out_norm (bool): Whether to use BN for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out):
        super(AttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.with_out = with_out
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm)

        self.value_project = self.build_project(
            key_in_channels,
            channels if self.with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm)

        if self.with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

    def build_project(self, in_channels: int , channels: int, num_convs: int, use_conv_module: bool):
        if use_conv_module:
            convs = [
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=1,
                    bias_attr=False)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvBNReLU(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias_attr=False))
        else:
            convs = [nn.Conv2D(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2D(channels, channels, 1))

        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats: paddle.Tensor, key_feats: paddle.Tensor) -> paddle.Tensor:
        query_shape = paddle.shape(query_feats)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.flatten(2).transpose([0, 2, 1])

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.flatten(2)
        value = value.flatten(2).transpose([0, 2, 1])
        sim_map = paddle.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.matmul(sim_map, value)
        context = paddle.transpose(context, [0, 2, 1])

        context = paddle.reshape(
            context, [0, self.out_channels, query_shape[2], query_shape[3]])

        if self.out_project is not None:
            context = self.out_project(context)
        return context
