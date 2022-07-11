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

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np
from paddlehub.module.module import moduleinfo
import paddlehub.vision.segmentation_transforms as T
from paddlehub.module.cv_module import ImageSegmentationModule
from paddleseg.utils import utils
from paddleseg.models import layers

from ginet_resnet50vd_cityscapes.resnet import ResNet50_vd


@moduleinfo(
    name="ginet_resnet50vd_cityscapes",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="GINetResnet50 is a segmentation model.",
    version="1.0.0",
    meta=ImageSegmentationModule)
class GINetResNet50(nn.Layer):
    """
    The GINetResNet50 implementation based on PaddlePaddle.
    The original article refers to
    Wu, Tianyi, Yu Lu, Yu Zhu, Chuang Zhang, Ming Wu, Zhanyu Ma, and Guodong Guo. "GINet: Graph interaction network for scene parsing." In European Conference on Computer Vision, pp. 34-51. Springer, Cham, 2020.
    (https://arxiv.org/pdf/2009.06160).
    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): Values in the tuple indicate the indices of output of backbone.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss.
            If true, auxiliary loss will be added after LearningToDownsample module. Default: False.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.. Default: False.
        jpu (bool, optional)): whether to use jpu unit in the base forward. Default:True.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes: int = 19,
                 backbone_indices: Tuple[int]=(0, 1, 2, 3),
                 enable_auxiliary_loss: bool = True,
                 align_corners: bool = True,
                 jpu: bool = True,
                 pretrained: str = None):
        super(GINetResNet50, self).__init__()
        self.nclass = num_classes
        self.aux = enable_auxiliary_loss
        self.jpu = jpu

        self.backbone = ResNet50_vd()
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners
        self.transforms = T.Compose([T.Normalize()])

        self.jpu = layers.JPU([512, 1024, 2048], width=512) if jpu else None
        self.head = GIHead(in_channels=2048, nclass=num_classes)

        if self.aux:
            self.auxlayer = layers.AuxLayer(
                1024, 1024 // 4, num_classes, bias_attr=False)

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

    def base_forward(self, x: paddle.Tensor) -> paddle.Tensor:
        feat_list = self.backbone(x)
        c1, c2, c3, c4 = [feat_list[i] for i in self.backbone_indices]

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        _, _, h, w = x.shape
        _, _, c3, c4 = self.base_forward(x)

        logit_list = []
        x, _ = self.head(c4)
        logit_list.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)

            logit_list.append(auxout)

        return [
            F.interpolate(
                logit, (h, w),
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]


class GIHead(nn.Layer):
    """The Graph Interaction Network head."""

    def __init__(self, in_channels: int, nclass: int):
        super().__init__()
        self.nclass = nclass
        inter_channels = in_channels // 4
        self.inp = paddle.zeros(shape=(nclass, 300), dtype='float32')
        self.inp = paddle.create_parameter(
            shape=self.inp.shape,
            dtype=str(self.inp.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.inp))

        self.fc1 = nn.Sequential(
            nn.Linear(300, 128), nn.BatchNorm1D(128), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1D(256), nn.ReLU())
        self.conv5 = layers.ConvBNReLU(
            in_channels,
            inter_channels,
            3,
            padding=1,
            bias_attr=False,
            stride=1)

        self.gloru = GlobalReasonUnit(
            in_channels=inter_channels,
            num_state=256,
            num_node=84,
            nclass=nclass)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1), nn.Conv2D(inter_channels, nclass, 1))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, C, H, W = x.shape
        inp = self.inp.detach()

        inp = self.fc1(inp)
        inp = self.fc2(inp).unsqueeze(axis=0).transpose((0, 2, 1))\
                           .expand((B, 256, self.nclass))

        out = self.conv5(x)

        out, se_out = self.gloru(out, inp)
        out = self.conv6(out)
        return out, se_out


class GlobalReasonUnit(nn.Layer):
    """
        The original paper refers to:
            Chen, Yunpeng, et al. "Graph-Based Global Reasoning Networks" (https://arxiv.org/abs/1811.12814)
    """

    def __init__(self, in_channels: int, num_state: int = 256, num_node: int = 84, nclass: int = 59):
        super().__init__()
        self.num_state = num_state
        self.conv_theta = nn.Conv2D(
            in_channels, num_node, kernel_size=1, stride=1, padding=0)
        self.conv_phi = nn.Conv2D(
            in_channels, num_state, kernel_size=1, stride=1, padding=0)
        self.graph = GraphLayer(num_state, num_node, nclass)
        self.extend_dim = nn.Conv2D(
            num_state, in_channels, kernel_size=1, bias_attr=False)

        self.bn = layers.SyncBatchNorm(in_channels)

    def forward(self, x: paddle.Tensor, inp: paddle.Tensor) -> paddle.Tensor:
        B = self.conv_theta(x)
        sizeB = B.shape
        B = B.reshape((sizeB[0], sizeB[1], -1))

        sizex = x.shape
        x_reduce = self.conv_phi(x)
        x_reduce = x_reduce.reshape((sizex[0], -1, sizex[2] * sizex[3]))\
                           .transpose((0, 2, 1))

        V = paddle.bmm(B, x_reduce).transpose((0, 2, 1))
        V = paddle.divide(
            V, paddle.to_tensor([sizex[2] * sizex[3]], dtype='float32'))

        class_node, new_V = self.graph(inp, V)
        D = B.reshape((sizeB[0], -1, sizeB[2] * sizeB[3])).transpose((0, 2, 1))
        Y = paddle.bmm(D, new_V.transpose((0, 2, 1)))
        Y = Y.transpose((0, 2, 1)).reshape((sizex[0], self.num_state, \
                                            sizex[2], -1))
        Y = self.extend_dim(Y)
        Y = self.bn(Y)
        out = Y + x

        return out, class_node


class GraphLayer(nn.Layer):
    def __init__(self, num_state: int, num_node: int, num_class: int):
        super().__init__()
        self.vis_gcn = GCN(num_state, num_node)
        self.word_gcn = GCN(num_state, num_class)
        self.transfer = GraphTransfer(num_state)
        self.gamma_vis = paddle.zeros([num_node])
        self.gamma_word = paddle.zeros([num_class])
        self.gamma_vis = paddle.create_parameter(
            shape=self.gamma_vis.shape,
            dtype=str(self.gamma_vis.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.gamma_vis))
        self.gamma_word = paddle.create_parameter(
            shape=self.gamma_word.shape,
            dtype=str(self.gamma_word.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.gamma_word))

    def forward(self, inp: paddle.Tensor, vis_node: paddle.Tensor) -> List[paddle.Tensor]:
        inp = self.word_gcn(inp)
        new_V = self.vis_gcn(vis_node)
        class_node, vis_node = self.transfer(inp, new_V)

        class_node = self.gamma_word * inp + class_node
        new_V = self.gamma_vis * vis_node + new_V
        return class_node, new_V


class GCN(nn.Layer):
    def __init__(self, num_state: int = 128, num_node: int = 64, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1D(
            num_node,
            num_node,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1D(
            num_state,
            num_state,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias_attr=bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h = self.conv1(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


class GraphTransfer(nn.Layer):
    """Transfer vis graph to class node, transfer class node to vis feature"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.channle_in = in_dim
        self.query_conv = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv_vis = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_word = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax_vis = nn.Softmax(axis=-1)
        self.softmax_word = nn.Softmax(axis=-2)

    def forward(self, word: paddle.Tensor, vis_node: paddle.Tensor) -> List[paddle.Tensor]:
        m_batchsize, C, Nc = word.shape
        m_batchsize, C, Nn = vis_node.shape

        proj_query = self.query_conv(word).reshape((m_batchsize, -1, Nc))\
                                          .transpose((0, 2, 1))
        proj_key = self.key_conv(vis_node).reshape((m_batchsize, -1, Nn))

        energy = paddle.bmm(proj_query, proj_key)
        attention_vis = self.softmax_vis(energy).transpose((0, 2, 1))
        attention_word = self.softmax_word(energy)

        proj_value_vis = self.value_conv_vis(vis_node).reshape((m_batchsize, -1,
                                                                Nn))
        proj_value_word = self.value_conv_word(word).reshape((m_batchsize, -1,
                                                              Nc))

        class_out = paddle.bmm(proj_value_vis, attention_vis)
        node_out = paddle.bmm(proj_value_word, attention_word)
        return class_out, node_out