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
from typing import Callable, Union, Tuple

import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import numpy as np
from paddlehub.module.module import moduleinfo
import paddlehub.vision.segmentation_transforms as T
from paddlehub.module.cv_module import ImageSegmentationModule

import fastscnn_cityscapes.layers as layers


@moduleinfo(
    name="fastscnn_cityscapes",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="fastscnn_cityscapes is a segmentation model.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class FastSCNN(nn.Layer):
    """
    The FastSCNN implementation based on PaddlePaddle.
    As mentioned in the original paper, FastSCNN is a real-time segmentation algorithm (123.5fps)
    even for high resolution images (1024x2048).
    The original article refers to
    Poudel, Rudra PK, et al. "Fast-scnn: Fast semantic segmentation network"
    (https://arxiv.org/pdf/1902.04502.pdf).
    Args:
        num_classes (int): The unique number of target classes, default is 19.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self, num_classes: int = 19, align_corners: bool = False, pretrained: str = None):

        super(FastSCNN, self).__init__()

        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=[64, 96, 128],
            out_channels=128,
            expansion=6,
            num_blocks=[3, 3, 3],
            align_corners=True)
        self.feature_fusion = FeatureFusionModule(64, 128, 128, align_corners)
        self.classifier = Classifier(128, num_classes)
        self.align_corners = align_corners
        self.transforms = T.Compose([T.Normalize()])

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'fastscnn_model.pdparams')
            model_dict = paddle.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")

    def transform(self, img: Union[np.ndarray, str]) -> Union[np.ndarray, str]:
        return self.transforms(img)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        logit_list = []
        input_size = paddle.shape(x)[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        logit = self.classifier(x)
        logit = F.interpolate(logit, input_size, mode='bilinear', align_corners=self.align_corners)
        logit_list.append(logit)

        return logit_list


class LearningToDownsample(nn.Layer):
    """
    Learning to downsample module.
    This module consists of three downsampling blocks (one conv and two separable conv)
    Args:
        dw_channels1 (int, optional): The input channels of the first sep conv. Default: 32.
        dw_channels2 (int, optional): The input channels of the second sep conv. Default: 48.
        out_channels (int, optional): The output channels of LearningToDownsample module. Default: 64.
    """

    def __init__(self, dw_channels1: int = 32, dw_channels2: int = 48, out_channels: int = 64):
        super(LearningToDownsample, self).__init__()

        self.conv_bn_relu = layers.ConvBNReLU(in_channels=3, out_channels=dw_channels1, kernel_size=3, stride=2)
        self.dsconv_bn_relu1 = layers.SeparableConvBNReLU(
            in_channels=dw_channels1, out_channels=dw_channels2, kernel_size=3, stride=2, padding=1)
        self.dsconv_bn_relu2 = layers.SeparableConvBNReLU(
            in_channels=dw_channels2, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv_bn_relu(x)
        x = self.dsconv_bn_relu1(x)
        x = self.dsconv_bn_relu2(x)
        return x


class GlobalFeatureExtractor(nn.Layer):
    """
    Global feature extractor module.
    This module consists of three InvertedBottleneck blocks (like inverted residual introduced by MobileNetV2) and
    a PPModule (introduced by PSPNet).
    Args:
        in_channels (int): The number of input channels to the module.
        block_channels (tuple): A tuple represents output channels of each bottleneck block.
        out_channels (int): The number of output channels of the module. Default:
        expansion (int): The expansion factor in bottleneck.
        num_blocks (tuple): It indicates the repeat time of each bottleneck.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, in_channels: int, block_channels: int, out_channels: int, expansion: int, num_blocks: Tuple[int],
                 align_corners: bool):
        super(GlobalFeatureExtractor, self).__init__()

        self.bottleneck1 = self._make_layer(InvertedBottleneck, in_channels, block_channels[0], num_blocks[0],
                                            expansion, 2)
        self.bottleneck2 = self._make_layer(InvertedBottleneck, block_channels[0], block_channels[1], num_blocks[1],
                                            expansion, 2)
        self.bottleneck3 = self._make_layer(InvertedBottleneck, block_channels[1], block_channels[2], num_blocks[2],
                                            expansion, 1)

        self.ppm = layers.PPModule(
            block_channels[2], out_channels, bin_sizes=(1, 2, 3, 6), dim_reduction=True, align_corners=align_corners)

    def _make_layer(self,
                    block: Callable,
                    in_channels: int,
                    out_channels: int,
                    blocks: int,
                    expansion: int = 6,
                    stride: int = 1):
        layers = []
        layers.append(block(in_channels, out_channels, expansion, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, expansion, 1))
        return nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class InvertedBottleneck(nn.Layer):
    """
    Single Inverted bottleneck implementation.
    Args:
        in_channels (int): The number of input channels to bottleneck block.
        out_channels (int): The number of output channels of bottleneck block.
        expansion (int, optional). The expansion factor in bottleneck. Default: 6.
        stride (int, optional). The stride used in depth-wise conv. Defalt: 2.
    """

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 6, stride: int = 2):
        super().__init__()

        self.use_shortcut = stride == 1 and in_channels == out_channels

        expand_channels = in_channels * expansion
        self.block = nn.Sequential(
            # pw
            layers.ConvBNReLU(in_channels=in_channels, out_channels=expand_channels, kernel_size=1, bias_attr=False),
            # dw
            layers.ConvBNReLU(
                in_channels=expand_channels,
                out_channels=expand_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expand_channels,
                bias_attr=False),
            # pw-linear
            layers.ConvBN(in_channels=expand_channels, out_channels=out_channels, kernel_size=1, bias_attr=False))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class FeatureFusionModule(nn.Layer):
    """
    Feature Fusion Module Implementation.
    This module fuses high-resolution feature and low-resolution feature.
    Args:
        high_in_channels (int): The channels of high-resolution feature (output of LearningToDownsample).
        low_in_channels (int): The channels of low-resolution feature (output of GlobalFeatureExtractor).
        out_channels (int): The output channels of this module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, high_in_channels: int, low_in_channels: int, out_channels: int, align_corners: bool):
        super().__init__()

        # Only depth-wise conv
        self.dwconv = layers.ConvBNReLU(
            in_channels=low_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=128,
            bias_attr=False)

        self.conv_low_res = layers.ConvBN(out_channels, out_channels, 1)
        self.conv_high_res = layers.ConvBN(high_in_channels, out_channels, 1)
        self.align_corners = align_corners

    def forward(self, high_res_input: int, low_res_input: int) -> paddle.Tensor:
        low_res_input = F.interpolate(
            low_res_input, paddle.shape(high_res_input)[2:], mode='bilinear', align_corners=self.align_corners)
        low_res_input = self.dwconv(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)
        high_res_input = self.conv_high_res(high_res_input)
        x = high_res_input + low_res_input

        return F.relu(x)


class Classifier(nn.Layer):
    """
    The Classifier module implementation.
    This module consists of two depth-wise conv and one conv.
    Args:
        input_channels (int): The input channels to this module.
        num_classes (int): The unique number of target classes.
    """

    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()

        self.dsconv1 = layers.SeparableConvBNReLU(
            in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1)

        self.dsconv2 = layers.SeparableConvBNReLU(
            in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1)

        self.conv = nn.Conv2D(in_channels=input_channels, out_channels=num_classes, kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)  # dropout_prob

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x
