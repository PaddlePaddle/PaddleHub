# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Union, List, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlehub.module.module import moduleinfo
import paddlehub.vision.segmentation_transforms as T
from paddlehub.module.cv_module import ImageSegmentationModule

from ann_resnet50_voc.resnet import ResNet50_vd
import ann_resnet50_voc.layers as layers

@moduleinfo(
    name="ann_resnet50_voc",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="ANNResnet50 is a segmentation model.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class ANN(nn.Layer):
    """
    The ANN implementation based on PaddlePaddle.

    The original article refers to
    Zhen, Zhu, et al. "Asymmetric Non-local Neural Networks for Semantic Segmentation"
    (https://arxiv.org/pdf/1908.07678.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        key_value_channels (int, optional): The key and value channels of self-attention map in both AFNB and APNB modules.
            Default: 256.
        inter_channels (int, optional): Both input and output channels of APNB modules. Default: 512.
        psp_size (tuple, optional): The out size of pooled feature maps. Default: (1, 3, 6, 8).
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes: int = 21,
                 backbone_indices: Tuple[int] = (2, 3),
                 key_value_channels: int = 256,
                 inter_channels: int = 512,
                 psp_size: Tuple[int] = (1, 3, 6, 8),
                 align_corners: bool = False,
                 pretrained: str = None):
        super(ANN, self).__init__()

        self.backbone = ResNet50_vd()
        backbone_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = ANNHead(num_classes, backbone_indices, backbone_channels,
                            key_value_channels, inter_channels, psp_size)
        self.align_corners = align_corners
        self.transforms = T.Compose([T.Normalize()])

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'model.pdparams')
            model_dict = paddle.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")
            
    def transform(self, img: Union[np.ndarray, str]) -> Union[np.ndarray, str]:
        return self.transforms(img)
    
    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]



class ANNHead(nn.Layer):
    """
    The ANNHead implementation.

    It mainly consists of AFNB and APNB modules.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            The first index will be taken as low-level features; the second one will be
            taken as high-level features in AFNB module. Usually backbone consists of four
            downsampling stage, such as ResNet, and return an output of each stage. If it is (2, 3),
            it means taking feature map of the third stage and the fourth stage in backbone.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        key_value_channels (int): The key and value channels of self-attention map in both AFNB and APNB modules.
        inter_channels (int): Both input and output channels of APNB modules.
        psp_size (tuple): The out size of pooled feature maps.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False
    """

    def __init__(self,
                 num_classes: int,
                 backbone_indices: Tuple[int],
                 backbone_channels: Tuple[int],
                 key_value_channels: int,
                 inter_channels: int,
                 psp_size: Tuple[int],
                 enable_auxiliary_loss: bool = False):
        super().__init__()

        low_in_channels = backbone_channels[0]
        high_in_channels = backbone_channels[1]

        self.fusion = AFNB(
            low_in_channels=low_in_channels,
            high_in_channels=high_in_channels,
            out_channels=high_in_channels,
            key_channels=key_value_channels,
            value_channels=key_value_channels,
            dropout_prob=0.05,
            repeat_sizes=([1]),
            psp_size=psp_size)

        self.context = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=high_in_channels,
                out_channels=inter_channels,
                kernel_size=3,
                padding=1),
            APNB(
                in_channels=inter_channels,
                out_channels=inter_channels,
                key_channels=key_value_channels,
                value_channels=key_value_channels,
                dropout_prob=0.05,
                repeat_sizes=([1]),
                psp_size=psp_size))

        self.cls = nn.Conv2D(
            in_channels=inter_channels, out_channels=num_classes, kernel_size=1)
        self.auxlayer = layers.AuxLayer(
            in_channels=low_in_channels,
            inter_channels=low_in_channels // 2,
            out_channels=num_classes,
            dropout_prob=0.05)

        self.backbone_indices = backbone_indices
        self.enable_auxiliary_loss = enable_auxiliary_loss

    def forward(self, feat_list: List[paddle.Tensor]) -> List[paddle.Tensor]:
        logit_list = []
        low_level_x = feat_list[self.backbone_indices[0]]
        high_level_x = feat_list[self.backbone_indices[1]]
        x = self.fusion(low_level_x, high_level_x)
        x = self.context(x)
        logit = self.cls(x)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            auxiliary_logit = self.auxlayer(low_level_x)
            logit_list.append(auxiliary_logit)

        return logit_list


class AFNB(nn.Layer):
    """
    Asymmetric Fusion Non-local Block.

    Args:
        low_in_channels (int): Low-level-feature channels.
        high_in_channels (int): High-level-feature channels.
        out_channels (int): Out channels of AFNB module.
        key_channels (int): The key channels in self-attention block.
        value_channels (int): The value channels in self-attention block.
        dropout_prob (float): The dropout rate of output.
        repeat_sizes (tuple, optional): The number of AFNB modules. Default: ([1]).
        psp_size (tuple. optional): The out size of pooled feature maps. Default: (1, 3, 6, 8).
    """

    def __init__(self,
                 low_in_channels: int,
                 high_in_channels: int,
                 out_channels: int,
                 key_channels: int,
                 value_channels: int,
                 dropout_prob: float,
                 repeat_sizes: Tuple[int] = ([1]),
                 psp_size: Tuple[int] = (1, 3, 6, 8)):
        super().__init__()

        self.psp_size = psp_size
        self.stages = nn.LayerList([
            SelfAttentionBlock_AFNB(low_in_channels, high_in_channels,
                                    key_channels, value_channels, out_channels,
                                    size) for size in repeat_sizes
        ])
        self.conv_bn = layers.ConvBN(
            in_channels=out_channels + high_in_channels,
            out_channels=out_channels,
            kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, low_feats: List[paddle.Tensor], high_feats: List[paddle.Tensor]) -> paddle.Tensor:
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]

        output = self.conv_bn(paddle.concat([context, high_feats], axis=1))
        output = self.dropout(output)

        return output


class APNB(nn.Layer):
    """
    Asymmetric Pyramid Non-local Block.

    Args:
        in_channels (int): The input channels of APNB module.
        out_channels (int): Out channels of APNB module.
        key_channels (int): The key channels in self-attention block.
        value_channels (int): The value channels in self-attention block.
        dropout_prob (float): The dropout rate of output.
        repeat_sizes (tuple, optional): The number of AFNB modules. Default: ([1]).
        psp_size (tuple, optional): The out size of pooled feature maps. Default: (1, 3, 6, 8).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key_channels: int,
                 value_channels: int,
                 dropout_prob: float,
                 repeat_sizes: Tuple[int] = ([1]),
                 psp_size: Tuple[int] = (1, 3, 6, 8)):
        super().__init__()

        self.psp_size = psp_size
        self.stages = nn.LayerList([
            SelfAttentionBlock_APNB(in_channels, out_channels, key_channels,
                                    value_channels, size)
            for size in repeat_sizes
        ])
        self.conv_bn = layers.ConvBNReLU(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        priors = [stage(x) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]

        output = self.conv_bn(paddle.concat([context, x], axis=1))
        output = self.dropout(output)

        return output


def _pp_module(x: paddle.Tensor, psp_size: List[int]) -> paddle.Tensor:
    n, c, h, w = x.shape
    priors = []
    for size in psp_size:
        feat = F.adaptive_avg_pool2d(x, size)
        feat = paddle.reshape(feat, shape=(0, c, -1))
        priors.append(feat)
    center = paddle.concat(priors, axis=-1)
    return center


class SelfAttentionBlock_AFNB(nn.Layer):
    """
    Self-Attention Block for AFNB module.

    Args:
        low_in_channels (int): Low-level-feature channels.
        high_in_channels (int): High-level-feature channels.
        key_channels (int): The key channels in self-attention block.
        value_channels (int): The value channels in self-attention block.
        out_channels (int, optional): Out channels of AFNB module. Default: None.
        scale (int, optional): Pooling size. Default: 1.
        psp_size (tuple, optional): The out size of pooled feature maps. Default: (1, 3, 6, 8).
    """

    def __init__(self,
                 low_in_channels: int,
                 high_in_channels: int,
                 key_channels: int,
                 value_channels: int,
                 out_channels: int = None,
                 scale: int = 1,
                 psp_size: Tuple[int] = (1, 3, 6, 8)):
        super().__init__()

        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool2D(scale)
        self.f_key = layers.ConvBNReLU(
            in_channels=low_in_channels,
            out_channels=key_channels,
            kernel_size=1)
        self.f_query = layers.ConvBNReLU(
            in_channels=high_in_channels,
            out_channels=key_channels,
            kernel_size=1)
        self.f_value = nn.Conv2D(
            in_channels=low_in_channels,
            out_channels=value_channels,
            kernel_size=1)

        self.W = nn.Conv2D(
            in_channels=value_channels,
            out_channels=out_channels,
            kernel_size=1)

        self.psp_size = psp_size

    def forward(self, low_feats: List[paddle.Tensor], high_feats: List[paddle.Tensor]) -> paddle.Tensor:
        batch_size, _, h, w = high_feats.shape

        value = self.f_value(low_feats)
        value = _pp_module(value, self.psp_size)
        value = paddle.transpose(value, (0, 2, 1))

        query = self.f_query(high_feats)
        query = paddle.reshape(query, shape=(0, self.key_channels, -1))
        query = paddle.transpose(query, perm=(0, 2, 1))

        key = self.f_key(low_feats)
        key = _pp_module(key, self.psp_size)

        sim_map = paddle.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.matmul(sim_map, value)
        context = paddle.transpose(context, perm=(0, 2, 1))
        hf_shape = paddle.shape(high_feats)
        context = paddle.reshape(
            context, shape=[0, self.value_channels, hf_shape[2], hf_shape[3]])

        context = self.W(context)

        return context


class SelfAttentionBlock_APNB(nn.Layer):
    """
    Self-Attention Block for APNB module.

    Args:
        in_channels (int): The input channels of APNB module.
        out_channels (int): The out channels of APNB module.
        key_channels (int): The key channels in self-attention block.
        value_channels (int): The value channels in self-attention block.
        scale (int, optional): Pooling size. Default: 1.
        psp_size (tuple, optional): The out size of pooled feature maps. Default: (1, 3, 6, 8).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key_channels: int,
                 value_channels: int,
                 scale: int = 1,
                 psp_size: Tuple[int] = (1, 3, 6, 8)):
        super().__init__()

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.pool = nn.MaxPool2D(scale)
        self.f_key = layers.ConvBNReLU(
            in_channels=self.in_channels,
            out_channels=self.key_channels,
            kernel_size=1)
        self.f_query = self.f_key
        self.f_value = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.value_channels,
            kernel_size=1)
        self.W = nn.Conv2D(
            in_channels=self.value_channels,
            out_channels=self.out_channels,
            kernel_size=1)

        self.psp_size = psp_size

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        batch_size, _, h, w = x.shape
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x)
        value = _pp_module(value, self.psp_size)
        value = paddle.transpose(value, perm=(0, 2, 1))

        query = self.f_query(x)
        query = paddle.reshape(query, shape=(0, self.key_channels, -1))
        query = paddle.transpose(query, perm=(0, 2, 1))

        key = self.f_key(x)
        key = _pp_module(key, self.psp_size)

        sim_map = paddle.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.matmul(sim_map, value)
        context = paddle.transpose(context, perm=(0, 2, 1))

        x_shape = paddle.shape(x)
        context = paddle.reshape(
            context, shape=[0, self.value_channels, x_shape[2], x_shape[3]])
        context = self.W(context)

        return context
