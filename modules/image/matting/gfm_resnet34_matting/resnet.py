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

import paddle
import paddle.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1,
    dilation: int=1) ->paddle.nn.Conv2D:
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, dilation=dilation, bias_attr=False)


def conv1x1(in_planes: int, out_planes: int, stride: int=1) ->paddle.nn.Conv2D:
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride,
        bias_attr=False)


class BasicBlock(nn.Layer):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1,
        downsample: Optional[nn.Layer]=None, groups: int=1, base_width:
        int=64, dilation: int=1, norm_layer: Optional[Callable[..., paddle.
        nn.Layer]]=None) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = paddle.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Layer):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1,
        downsample: Optional[nn.Layer]=None, groups: int=1, base_width:
        int=64, dilation: int=1, norm_layer: Optional[Callable[..., paddle.
        nn.Layer]]=None) ->None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Layer):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers:
        List[int], num_classes: int=1000, zero_init_residual: bool=False,
        groups: int=1, width_per_group: int=64,
        replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer:
        Optional[Callable[..., paddle.nn.Layer]]=None) ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = paddle.nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
        planes: int, blocks: int, stride: int=1, dilate: bool=False
        ) ->paddle.nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x= paddle.flatten(x,1)
        x = self.fc(x)
        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self._forward_impl(x)


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers:
    List[int], pretrained: bool, progress: bool, **kwargs: Any) ->ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet34(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained,
        progress, **kwargs)
