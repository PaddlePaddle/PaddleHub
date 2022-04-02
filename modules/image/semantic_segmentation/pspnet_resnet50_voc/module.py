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

from pspnet_resnet50_voc.resnet import ResNet50_vd
import pspnet_resnet50_voc.layers as layers

@moduleinfo(
    name="pspnet_resnet50_voc",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="PSPNetResnet50 is a segmentation model.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class PSPNet(nn.Layer):
    """
    The PSPNet implementation based on PaddlePaddle.

    The original article refers to
    Zhao, Hengshuang, et al. "Pyramid scene parsing network"
    (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        pp_out_channels (int, optional): The output channels after Pyramid Pooling Module. Default: 1024.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1,2,3,6).
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes: int = 21,
                 backbone_indices: Tuple[int] = (2, 3),
                 pp_out_channels: int = 1024,
                 bin_sizes: Tuple[int] = (1, 2, 3, 6),
                 enable_auxiliary_loss: bool  = True,
                 align_corners: bool = False,
                 pretrained: str = None):
        super(PSPNet, self).__init__()

        self.backbone = ResNet50_vd()
        backbone_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = PSPNetHead(num_classes, backbone_indices, backbone_channels,
                               pp_out_channels, bin_sizes,
                               enable_auxiliary_loss, align_corners)
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

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]


class PSPNetHead(nn.Layer):
    """
    The PSPNetHead implementation.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            The first index will be taken as a deep-supervision feature in auxiliary layer;
            the second one will be taken as input of Pyramid Pooling Module (PPModule).
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (2, 3) in ResNet, that means taking feature map of the third
            stage (res4b22) in backbone, and feature map of the fourth stage (res5c) as input of PPModule.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        pp_out_channels (int): The output channels after Pyramid Pooling Module.
        bin_sizes (tuple): The out size of pooled feature maps.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 pp_out_channels, bin_sizes, enable_auxiliary_loss,
                 align_corners):

        super().__init__()

        self.backbone_indices = backbone_indices

        self.psp_module = layers.PPModule(
            in_channels=backbone_channels[1],
            out_channels=pp_out_channels,
            bin_sizes=bin_sizes,
            dim_reduction=True,
            align_corners=align_corners)

        self.dropout = nn.Dropout(p=0.1)  # dropout_prob

        self.conv = nn.Conv2D(
            in_channels=pp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        if enable_auxiliary_loss:
            self.auxlayer = layers.AuxLayer(
                in_channels=backbone_channels[0],
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss

    def forward(self, feat_list: List[paddle.Tensor]) -> List[paddle.Tensor]:
        logit_list = []
        x = feat_list[self.backbone_indices[1]]
        x = self.psp_module(x)
        x = self.dropout(x)
        logit = self.conv(x)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            auxiliary_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(auxiliary_feat)
            logit_list.append(auxiliary_logit)

        return logit_list
