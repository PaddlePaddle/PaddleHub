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

import os
from typing import Union, List, Tuple

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np
from paddlehub.module.module import moduleinfo
import paddlehub.vision.segmentation_transforms as T
from paddlehub.module.cv_module import ImageSegmentationModule

from deeplabv3p_resnet50_cityscapes.resnet import ResNet50_vd
import deeplabv3p_resnet50_cityscapes.layers as L


@moduleinfo(
    name="deeplabv3p_resnet50_cityscapes",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="DeepLabV3PResnet50 is a segmentation model.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class DeepLabV3PResnet50(nn.Layer):
    """
    The DeepLabV3PResnet50 implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        num_classes (int): the unique number of target classes.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage, so we set default (0, 3), which means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        aspp_ratios (tuple): the dilation rate using in ASSP module.
            if output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            if output_stride=8, aspp_ratios is (1, 12, 24, 36).
        aspp_out_channels (int): the output channels of ASPP module.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str): the path of pretrained model. Default to None.
    """

    def __init__(self,
                 num_classes: int = 19,
                 backbone_indices: Tuple[int] = (0, 3),
                 aspp_ratios: Tuple[int] = (1, 12, 24, 36),
                 aspp_out_channels: int = 256,
                 align_corners=False,
                 pretrained: str = None):
        super(DeepLabV3PResnet50, self).__init__()
        self.backbone = ResNet50_vd()
        backbone_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.head = DeepLabV3PHead(num_classes, backbone_indices, backbone_channels, aspp_ratios, aspp_out_channels,
                                   align_corners)
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
            F.interpolate(logit, x.shape[2:], mode='bilinear', align_corners=self.align_corners) for logit in logit_list
        ]


class DeepLabV3PHead(nn.Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, num_classes: int, backbone_indices: Tuple[paddle.Tensor],
                 backbone_channels: Tuple[paddle.Tensor], aspp_ratios: Tuple[float], aspp_out_channels: int,
                 align_corners: bool):
        super().__init__()

        self.aspp = L.ASPPModule(
            aspp_ratios, backbone_channels[1], aspp_out_channels, align_corners, use_sep_conv=True, image_pooling=True)
        self.decoder = Decoder(num_classes, backbone_channels[0], align_corners)
        self.backbone_indices = backbone_indices

    def forward(self, feat_list: List[paddle.Tensor]) -> List[paddle.Tensor]:
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)
        return logit_list


class Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, num_classes: int, in_channels: int, align_corners: bool):
        super(Decoder, self).__init__()

        self.conv_bn_relu1 = L.ConvBNReLU(in_channels=in_channels, out_channels=48, kernel_size=1)

        self.conv_bn_relu2 = L.SeparableConvBNReLU(in_channels=304, out_channels=256, kernel_size=3, padding=1)
        self.conv_bn_relu3 = L.SeparableConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv = nn.Conv2D(in_channels=256, out_channels=num_classes, kernel_size=1)

        self.align_corners = align_corners

    def forward(self, x: paddle.Tensor, low_level_feat: paddle.Tensor) -> paddle.Tensor:
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        x = F.interpolate(x, low_level_feat.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x = paddle.concat([x, low_level_feat], axis=1)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x
