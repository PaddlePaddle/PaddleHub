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
from typing import Union, List, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlehub.module.module import moduleinfo
import paddlehub.vision.segmentation_transforms as T
from paddlehub.module.cv_module import ImageSegmentationModule

import bisenet_cityscapes.layers as layers


@moduleinfo(
    name="bisenetv2_cityscapes",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="Bisenet is a segmentation model trained by Cityscapes.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class BiSeNetV2(nn.Layer):
    """
    The BiSeNet V2 implementation based on PaddlePaddle.

    The original article refers to
    Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
    (https://arxiv.org/abs/2004.02147)

    Args:
        num_classes (int): The unique number of target classes, default is 19.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self, num_classes: int = 19, lambd: float = 0.25, align_corners: bool = False, pretrained: str = None):
        super(BiSeNetV2, self).__init__()

        C1, C2, C3 = 64, 64, 128
        db_channels = (C1, C2, C3)
        C1, C3, C4, C5 = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = (C1, C3, C4, C5)
        mid_channels = 128

        self.db = DetailBranch(db_channels)
        self.sb = SemanticBranch(sb_channels)

        self.bga = BGA(mid_channels, align_corners)
        self.aux_head1 = SegHead(C1, C1, num_classes)
        self.aux_head2 = SegHead(C3, C3, num_classes)
        self.aux_head3 = SegHead(C4, C4, num_classes)
        self.aux_head4 = SegHead(C5, C5, num_classes)
        self.head = SegHead(mid_channels, mid_channels, num_classes)

        self.align_corners = align_corners
        self.transforms = T.Compose([T.Normalize()])

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'bisenet_model.pdparams')
            model_dict = paddle.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")

    def transform(self, img: Union[np.ndarray, str]) -> Union[np.ndarray, str]:
        return self.transforms(img)

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        logit = self.head(self.bga(dfm, sfm))

        if not self.training:
            logit_list = [logit]
        else:
            logit1 = self.aux_head1(feat1)
            logit2 = self.aux_head2(feat2)
            logit3 = self.aux_head3(feat3)
            logit4 = self.aux_head4(feat4)
            logit_list = [logit, logit1, logit2, logit3, logit4]

        logit_list = [
            F.interpolate(logit, paddle.shape(x)[2:], mode='bilinear', align_corners=self.align_corners)
            for logit in logit_list
        ]

        return logit_list


class StemBlock(nn.Layer):
    def __init__(self, in_dim: int, out_dim: int):
        super(StemBlock, self).__init__()

        self.conv = layers.ConvBNReLU(in_dim, out_dim, 3, stride=2)

        self.left = nn.Sequential(
            layers.ConvBNReLU(out_dim, out_dim // 2, 1), layers.ConvBNReLU(out_dim // 2, out_dim, 3, stride=2))

        self.right = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.fuse = layers.ConvBNReLU(out_dim * 2, out_dim, 3)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        concat = paddle.concat([left, right], axis=1)
        return self.fuse(concat)


class ContextEmbeddingBlock(nn.Layer):
    def __init__(self, in_dim: int, out_dim: int):
        super(ContextEmbeddingBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.bn = layers.SyncBatchNorm(in_dim)

        self.conv_1x1 = layers.ConvBNReLU(in_dim, out_dim, 1)
        self.conv_3x3 = nn.Conv2D(out_dim, out_dim, 3, 1, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        gap = self.gap(x)
        bn = self.bn(gap)
        conv1 = self.conv_1x1(bn) + x
        return self.conv_3x3(conv1)


class GatherAndExpansionLayer1(nn.Layer):
    """Gather And Expansion Layer with stride 1"""

    def __init__(self, in_dim: int, out_dim: int, expand: int):
        super().__init__()

        expand_dim = expand * in_dim

        self.conv = nn.Sequential(
            layers.ConvBNReLU(in_dim, in_dim, 3), layers.DepthwiseConvBN(in_dim, expand_dim, 3),
            layers.ConvBN(expand_dim, out_dim, 1))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.relu(self.conv(x) + x)


class GatherAndExpansionLayer2(nn.Layer):
    """Gather And Expansion Layer with stride 2"""

    def __init__(self, in_dim: int, out_dim: int, expand: int):
        super().__init__()

        expand_dim = expand * in_dim

        self.branch_1 = nn.Sequential(
            layers.ConvBNReLU(in_dim, in_dim, 3), layers.DepthwiseConvBN(in_dim, expand_dim, 3, stride=2),
            layers.DepthwiseConvBN(expand_dim, expand_dim, 3), layers.ConvBN(expand_dim, out_dim, 1))

        self.branch_2 = nn.Sequential(
            layers.DepthwiseConvBN(in_dim, in_dim, 3, stride=2), layers.ConvBN(in_dim, out_dim, 1))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.relu(self.branch_1(x) + self.branch_2(x))


class DetailBranch(nn.Layer):
    """The detail branch of BiSeNet, which has wide channels but shallow layers."""

    def __init__(self, in_channels: int):
        super().__init__()

        C1, C2, C3 = in_channels

        self.convs = nn.Sequential(
            # stage 1
            layers.ConvBNReLU(3, C1, 3, stride=2),
            layers.ConvBNReLU(C1, C1, 3),
            # stage 2
            layers.ConvBNReLU(C1, C2, 3, stride=2),
            layers.ConvBNReLU(C2, C2, 3),
            layers.ConvBNReLU(C2, C2, 3),
            # stage 3
            layers.ConvBNReLU(C2, C3, 3, stride=2),
            layers.ConvBNReLU(C3, C3, 3),
            layers.ConvBNReLU(C3, C3, 3),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.convs(x)


class SemanticBranch(nn.Layer):
    """The semantic branch of BiSeNet, which has narrow channels but deep layers."""

    def __init__(self, in_channels: int):
        super().__init__()
        C1, C3, C4, C5 = in_channels

        self.stem = StemBlock(3, C1)

        self.stage3 = nn.Sequential(GatherAndExpansionLayer2(C1, C3, 6), GatherAndExpansionLayer1(C3, C3, 6))

        self.stage4 = nn.Sequential(GatherAndExpansionLayer2(C3, C4, 6), GatherAndExpansionLayer1(C4, C4, 6))

        self.stage5_4 = nn.Sequential(
            GatherAndExpansionLayer2(C4, C5, 6), GatherAndExpansionLayer1(C5, C5, 6), GatherAndExpansionLayer1(
                C5, C5, 6), GatherAndExpansionLayer1(C5, C5, 6))

        self.ce = ContextEmbeddingBlock(C5, C5)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        stage2 = self.stem(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5_4 = self.stage5_4(stage4)
        fm = self.ce(stage5_4)
        return stage2, stage3, stage4, stage5_4, fm


class BGA(nn.Layer):
    """The Bilateral Guided Aggregation Layer, used to fuse the semantic features and spatial features."""

    def __init__(self, out_dim: int, align_corners: bool):
        super().__init__()

        self.align_corners = align_corners

        self.db_branch_keep = nn.Sequential(layers.DepthwiseConvBN(out_dim, out_dim, 3), nn.Conv2D(out_dim, out_dim, 1))

        self.db_branch_down = nn.Sequential(
            layers.ConvBN(out_dim, out_dim, 3, stride=2), nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

        self.sb_branch_keep = nn.Sequential(
            layers.DepthwiseConvBN(out_dim, out_dim, 3), nn.Conv2D(out_dim, out_dim, 1),
            layers.Activation(act='sigmoid'))

        self.sb_branch_up = layers.ConvBN(out_dim, out_dim, 3)

        self.conv = layers.ConvBN(out_dim, out_dim, 3)

    def forward(self, dfm: int, sfm: int) -> paddle.Tensor:
        db_feat_keep = self.db_branch_keep(dfm)
        db_feat_down = self.db_branch_down(dfm)
        sb_feat_keep = self.sb_branch_keep(sfm)

        sb_feat_up = self.sb_branch_up(sfm)
        sb_feat_up = F.interpolate(
            sb_feat_up, paddle.shape(db_feat_keep)[2:], mode='bilinear', align_corners=self.align_corners)

        sb_feat_up = F.sigmoid(sb_feat_up)
        db_feat = db_feat_keep * sb_feat_up

        sb_feat = db_feat_down * sb_feat_keep
        sb_feat = F.interpolate(sb_feat, paddle.shape(db_feat)[2:], mode='bilinear', align_corners=self.align_corners)

        return self.conv(db_feat + sb_feat)


class SegHead(nn.Layer):
    def __init__(self, in_dim: int, mid_dim: int, num_classes: int):
        super().__init__()

        self.conv_3x3 = nn.Sequential(layers.ConvBNReLU(in_dim, mid_dim, 3), nn.Dropout(0.1))

        self.conv_1x1 = nn.Conv2D(mid_dim, num_classes, 1, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        return conv2
