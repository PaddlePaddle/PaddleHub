# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from danet_resnet50_voc.resnet import ResNet50_vd
import danet_resnet50_voc.layers as L


@moduleinfo(
    name="danet_resnet50_cityscapes",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="DANetResnet50 is a segmentation model.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class DANet(nn.Layer):
    """
    The DANet implementation based on PaddlePaddle.

    The original article refers to
    Fu, jun, et al. "Dual Attention Network for Scene Segmentation"
    (https://arxiv.org/pdf/1809.02983.pdf)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of
            output of backbone.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes: int = 19,
                 backbone_indices: Tuple[int] = (2, 3),
                 align_corners: bool = False,
                 pretrained: str = None):
        super(DANet, self).__init__()

        self.backbone = ResNet50_vd()
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]

        self.head = DAHead(num_classes=num_classes, in_channels=in_channels)

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

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        if not self.training:
            logit_list = [logit_list[0]]

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                align_mode=1) for logit in logit_list
        ]
        return logit_list

    def transform(self, img: Union[np.ndarray, str]) -> Union[np.ndarray, str]:
        return self.transforms(img)


class DAHead(nn.Layer):
    """
    The Dual attention head.

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (tuple): The number of input channels.
    """

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        in_channels = in_channels[-1]
        inter_channels = in_channels // 4

        self.channel_conv = L.ConvBNReLU(in_channels, inter_channels, 3)
        self.position_conv = L.ConvBNReLU(in_channels, inter_channels, 3)
        self.pam = PAM(inter_channels)
        self.cam = CAM(inter_channels)
        self.conv1 = L.ConvBNReLU(inter_channels, inter_channels, 3)
        self.conv2 = L.ConvBNReLU(inter_channels, inter_channels, 3)

        self.aux_head = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(in_channels, num_classes, 1))

        self.aux_head_pam = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(inter_channels, num_classes, 1))

        self.aux_head_cam = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(inter_channels, num_classes, 1))

        self.cls_head = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(inter_channels, num_classes, 1))

    def forward(self, feat_list: List[paddle.Tensor]) -> List[paddle.Tensor]:
        feats = feat_list[-1]
        channel_feats = self.channel_conv(feats)
        channel_feats = self.cam(channel_feats)
        channel_feats = self.conv1(channel_feats)

        position_feats = self.position_conv(feats)
        position_feats = self.pam(position_feats)
        position_feats = self.conv2(position_feats)

        feats_sum = position_feats + channel_feats
        logit = self.cls_head(feats_sum)

        if not self.training:
            return [logit]

        cam_logit = self.aux_head_cam(channel_feats)
        pam_logit = self.aux_head_cam(position_feats)
        aux_logit = self.aux_head(feats)
        return [logit, cam_logit, pam_logit, aux_logit]


class PAM(nn.Layer):
    """Position attention module."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 8
        self.mid_channels = mid_channels
        self.in_channels = in_channels

        self.query_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.key_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.value_conv = nn.Conv2D(in_channels, in_channels, 1, 1)

        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x_shape = paddle.shape(x)

        # query: n, h * w, c1
        query = self.query_conv(x)
        query = paddle.reshape(query, (0, self.mid_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        # key: n, c1, h * w
        key = self.key_conv(x)
        key = paddle.reshape(key, (0, self.mid_channels, -1))

        # sim: n, h * w, h * w
        sim = paddle.bmm(query, key)
        sim = F.softmax(sim, axis=-1)

        value = self.value_conv(x)
        value = paddle.reshape(value, (0, self.in_channels, -1))
        sim = paddle.transpose(sim, (0, 2, 1))

        # feat: from (n, c2, h * w) -> (n, c2, h, w)
        feat = paddle.bmm(value, sim)
        feat = paddle.reshape(feat,
                              (0, self.in_channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out


class CAM(nn.Layer):
    """Channel attention module."""

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels
        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x_shape = paddle.shape(x)
        # query: n, c, h * w
        query = paddle.reshape(x, (0, self.channels, -1))
        # key: n, h * w, c
        key = paddle.reshape(x, (0, self.channels, -1))
        key = paddle.transpose(key, (0, 2, 1))

        # sim: n, c, c
        sim = paddle.bmm(query, key)
        # The danet author claims that this can avoid gradient divergence
        sim = paddle.max(
            sim, axis=-1, keepdim=True).tile([1, 1, self.channels]) - sim
        sim = F.softmax(sim, axis=-1)

        # feat: from (n, c, h * w) to (n, c, h, w)
        value = paddle.reshape(x, (0, self.channels, -1))
        feat = paddle.bmm(sim, value)
        feat = paddle.reshape(feat, (0, self.channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out
