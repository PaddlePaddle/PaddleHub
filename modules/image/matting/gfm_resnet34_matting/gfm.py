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

from typing import Callable, Union, List, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from gfm_resnet34_matting.resnet import resnet34


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> Callable:
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias_attr=False)


def conv_up_psp(in_channels: int, out_channels: int, up_sample: float) -> Callable:
    return nn.Sequential(nn.Conv2D(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2D(out_channels), 
        nn.ReLU(),
        nn.Upsample(scale_factor=up_sample, mode='bilinear',align_corners = False))


def build_bb(in_channels: int, mid_channels: int, out_channels: int) -> Callable:
    return nn.Sequential(nn.Conv2D(in_channels, mid_channels, 3, dilation=2,
        padding=2), nn.BatchNorm2D(mid_channels), nn.
        ReLU(), nn.Conv2D(mid_channels, out_channels, 3,
        dilation=2, padding=2), nn.BatchNorm2D(out_channels), nn.ReLU(), nn.Conv2D(out_channels,
        out_channels, 3, dilation=2, padding=2), nn.BatchNorm2D(
        out_channels), nn.ReLU())


def build_decoder(in_channels: int, mid_channels_1: int, mid_channels_2: int, out_channels: int,
    last_bnrelu: bool, upsample_flag: bool) -> Callable:
    layers = []
    layers += [nn.Conv2D(in_channels, mid_channels_1, 3, padding=1), nn.
        BatchNorm2D(mid_channels_1), nn.ReLU(), nn.Conv2D(mid_channels_1, mid_channels_2, 3, padding=1), nn.
        BatchNorm2D(mid_channels_2), nn.ReLU(), nn.Conv2D(mid_channels_2, out_channels, 3, padding=1)]
    if last_bnrelu:
        layers += [nn.BatchNorm2D(out_channels), nn.ReLU()]
    
    if upsample_flag:
        layers += [nn.Upsample(scale_factor=2, mode='bilinear')]

    sequential = nn.Sequential(*layers)
    return sequential


class BasicBlock(nn.Layer):
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: paddle.Tensor) -> Callable:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PSPModule(nn.Layer):

    def __init__(self, features: paddle.Tensor, out_features: int = 1024, sizes: List[int] = (1, 2, 3, 6)):
        super().__init__()
        #self.stages = []
        self.stages = nn.LayerList([self._make_stage(features, size) for
            size in sizes])
        self.bottleneck = nn.Conv2D(features * (len(sizes) + 1),
            out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features: paddle.Tensor, size: int) -> Callable:
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = nn.Conv2D(features, features, kernel_size=1, bias_attr=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats: paddle.Tensor) -> paddle.Tensor:
        h, w = feats.shape[2], feats.shape[3]
        priors = [F.upsample(stage(feats), size=(h, w), mode='bilinear',align_corners = True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(paddle.concat(priors, 1))
        return self.relu(bottle)


class SELayer(nn.Layer):

    def __init__(self, channel: int, reduction: int = 4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction,
            bias_attr=False), nn.ReLU(), nn.
            Linear(channel // reduction, channel, bias_attr=False), nn.
            Sigmoid())

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GFM(nn.Layer):
    """
    The GFM implementation based on PaddlePaddle.
    
    The original article refers to：
    Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
    Main network file (GFM).
    
    Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
    Licensed under the MIT License (see LICENSE for details)
    Github repo: https://github.com/JizhiziLi/GFM
    Paper link (Arxiv): https://arxiv.org/abs/2010.16188

    """

    def __init__(self):
        super().__init__()
        self.backbone = 'r34_2b'
        self.rosta = 'TT'
        if self.rosta == 'TT':
            self.gd_channel = 3
        else:
            self.gd_channel = 2
        if self.backbone == 'r34_2b':
            self.resnet = resnet34()
            self.encoder0 = nn.Sequential(nn.Conv2D(3, 64, 3, padding=1),
                nn.BatchNorm2D(64), nn.ReLU())
            self.encoder1 = self.resnet.layer1
            self.encoder2 = self.resnet.layer2
            self.encoder3 = self.resnet.layer3
            self.encoder4 = self.resnet.layer4
            self.encoder5 = nn.Sequential(nn.MaxPool2D(2, 2, ceil_mode=True
                ), BasicBlock(512, 512), BasicBlock(512, 512), BasicBlock(
                512, 512))
            self.encoder6 = nn.Sequential(nn.MaxPool2D(2, 2, ceil_mode=True
                ), BasicBlock(512, 512), BasicBlock(512, 512), BasicBlock(
                512, 512))
            self.psp_module = PSPModule(512, 512, (1, 3, 5))
            self.psp6 = conv_up_psp(512, 512, 2)
            self.psp5 = conv_up_psp(512, 512, 4)
            self.psp4 = conv_up_psp(512, 256, 8)
            self.psp3 = conv_up_psp(512, 128, 16)
            self.psp2 = conv_up_psp(512, 64, 32)
            self.psp1 = conv_up_psp(512, 64, 32)
            self.decoder6_g = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder5_g = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder4_g = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_g = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_g = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_g = build_decoder(128, 64, 64, 64, True, False)
            self.bridge_block = build_bb(512, 512, 512)
            self.decoder6_f = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder5_f = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_f = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_f = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_f = build_decoder(128, 64, 64, 64, True, False)
            if self.rosta == 'RIM':
                self.decoder0_g_tt = nn.Sequential(nn.Conv2D(64, 3, 3,
                    padding=1))
                self.decoder0_g_ft = nn.Sequential(nn.Conv2D(64, 2, 3,
                    padding=1))
                self.decoder0_g_bt = nn.Sequential(nn.Conv2D(64, 2, 3,
                    padding=1))
                self.decoder0_f_tt = nn.Sequential(nn.Conv2D(64, 1, 3,
                    padding=1))
                self.decoder0_f_ft = nn.Sequential(nn.Conv2D(64, 1, 3,
                    padding=1))
                self.decoder0_f_bt = nn.Sequential(nn.Conv2D(64, 1, 3,
                    padding=1))
            else:
                self.decoder0_g = nn.Sequential(nn.Conv2D(64, self.
                    gd_channel, 3, padding=1))
                self.decoder0_f = nn.Sequential(nn.Conv2D(64, 1, 3, padding=1))
        if self.backbone == 'r34':
            self.encoder0 = nn.Sequential(self.resnet.conv1, self.resnet.
                bn1, self.resnet.relu)
            
            self.encoder1 = nn.Sequential(self.resnet.maxpool, self.resnet.
                layer1)
            self.encoder2 = self.resnet.layer2
            self.encoder3 = self.resnet.layer3
            self.encoder4 = self.resnet.layer4
            self.psp_module = PSPModule(512, 512, (1, 3, 5))
            self.psp4 = conv_up_psp(512, 256, 2)
            self.psp3 = conv_up_psp(512, 128, 4)
            self.psp2 = conv_up_psp(512, 64, 8)
            self.psp1 = conv_up_psp(512, 64, 16)
            self.decoder4_g = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_g = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_g = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_g = build_decoder(128, 64, 64, 64, True, True)
            self.bridge_block = build_bb(512, 512, 512)
            self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_f = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_f = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_f = build_decoder(128, 64, 64, 64, True, True)
            if self.rosta == 'RIM':
                self.decoder0_g_tt = build_decoder(128, 64, 64, 3, False, True)
                self.decoder0_g_ft = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_g_bt = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_f_tt = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_ft = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_bt = build_decoder(128, 64, 64, 1, False, True)
            else:
                self.decoder0_g = build_decoder(128, 64, 64, self.
                    gd_channel, False, True)
                self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)
        elif self.backbone == 'r101':
            self.encoder0 = nn.Sequential(self.resnet.conv1, self.resnet.
                bn1, self.resnet.relu)
            self.encoder1 = nn.Sequential(self.resnet.maxpool, self.resnet.
                layer1)
            self.encoder2 = self.resnet.layer2
            self.encoder3 = self.resnet.layer3
            self.encoder4 = self.resnet.layer4
            self.psp_module = PSPModule(2048, 2048, (1, 3, 5))
            self.bridge_block = build_bb(2048, 2048, 2048)
            self.psp4 = conv_up_psp(2048, 1024, 2)
            self.psp3 = conv_up_psp(2048, 512, 4)
            self.psp2 = conv_up_psp(2048, 256, 8)
            self.psp1 = conv_up_psp(2048, 64, 16)
            self.decoder4_g = build_decoder(4096, 2048, 1024, 1024, True, True)
            self.decoder3_g = build_decoder(2048, 1024, 512, 512, True, True)
            self.decoder2_g = build_decoder(1024, 512, 256, 256, True, True)
            self.decoder1_g = build_decoder(512, 256, 128, 64, True, True)
            self.decoder4_f = build_decoder(4096, 2048, 1024, 1024, True, True)
            self.decoder3_f = build_decoder(2048, 1024, 512, 512, True, True)
            self.decoder2_f = build_decoder(1024, 512, 256, 256, True, True)
            self.decoder1_f = build_decoder(512, 256, 128, 64, True, True)
            if self.rosta == 'RIM':
                self.decoder0_g_tt = build_decoder(128, 64, 64, 3, False, True)
                self.decoder0_g_ft = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_g_bt = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_f_tt = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_ft = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_bt = build_decoder(128, 64, 64, 1, False, True)
            else:
                self.decoder0_g = build_decoder(128, 64, 64, self.
                    gd_channel, False, True)
                self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)
        elif self.backbone == 'd121':
            self.encoder0 = nn.Sequential(self.densenet.features.conv0,
                self.densenet.features.norm0, self.densenet.features.relu0)
            self.encoder1 = nn.Sequential(self.densenet.features.
                denseblock1, self.densenet.features.transition1)
            self.encoder2 = nn.Sequential(self.densenet.features.
                denseblock2, self.densenet.features.transition2)
            self.encoder3 = nn.Sequential(self.densenet.features.
                denseblock3, self.densenet.features.transition3)
            self.encoder4 = nn.Sequential(self.densenet.features.
                denseblock4, nn.Conv2D(1024, 512, 3, padding=1), nn.
                BatchNorm2D(512), nn.ReLU(),
                nn.MaxPool2D(2, 2, ceil_mode=True))
            self.psp_module = PSPModule(512, 512, (1, 3, 5))
            self.psp4 = conv_up_psp(512, 256, 2)
            self.psp3 = conv_up_psp(512, 128, 4)
            self.psp2 = conv_up_psp(512, 64, 8)
            self.psp1 = conv_up_psp(512, 64, 16)
            self.decoder4_g = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_g = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_g = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_g = build_decoder(128, 64, 64, 64, True, True)
            self.bridge_block = build_bb(512, 512, 512)
            self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_f = build_decoder(768, 256, 256, 128, True, True)
            self.decoder2_f = build_decoder(384, 128, 128, 64, True, True)
            self.decoder1_f = build_decoder(192, 64, 64, 64, True, True)
            if self.rosta == 'RIM':
                self.decoder0_g_tt = build_decoder(128, 64, 64, 3, False, True)
                self.decoder0_g_ft = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_g_bt = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_f_tt = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_ft = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_bt = build_decoder(128, 64, 64, 1, False, True)
            else:
                self.decoder0_g = build_decoder(128, 64, 64, self.
                    gd_channel, False, True)
                self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)
        if self.rosta == 'RIM':
            self.rim = nn.Sequential(nn.Conv2D(3, 16, 1), SELayer(16), nn.
                Conv2D(16, 1, 1))

    def forward(self, input: paddle.Tensor) -> List[paddle.Tensor]:
        glance_sigmoid = paddle.zeros(input.shape)
        glance_sigmoid.stop_gradient = True
        focus_sigmoid = paddle.zeros(input.shape)
        focus_sigmoid.stop_gradient = True
        fusion_sigmoid = paddle.zeros(input.shape)
        fusion_sigmoid.stop_gradient = True
        e0 = self.encoder0(input)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        if self.backbone == 'r34_2b':
            e5 = self.encoder5(e4)
            e6 = self.encoder6(e5)
            psp = self.psp_module(e6)
            d6_g = self.decoder6_g(paddle.concat((psp, e6), 1))
            d5_g = self.decoder5_g(paddle.concat((self.psp6(psp),
                d6_g), 1))
            d4_g = self.decoder4_g(paddle.concat((self.psp5(psp),
                d5_g), 1))
        else:
            psp = self.psp_module(e4)
            d4_g = self.decoder4_g(paddle.concat((psp, e4), 1))
        d3_g = self.decoder3_g(paddle.concat((self.psp4(psp), d4_g), 1))
        d2_g = self.decoder2_g(paddle.concat((self.psp3(psp), d3_g), 1))
        d1_g = self.decoder1_g(paddle.concat((self.psp2(psp), d2_g), 1))
        if self.backbone == 'r34_2b':
            if self.rosta == 'RIM':
                d0_g_tt = self.decoder0_g_tt(d1_g)
                d0_g_ft = self.decoder0_g_ft(d1_g)
                d0_g_bt = self.decoder0_g_bt(d1_g)
            else:
                d0_g = self.decoder0_g(d1_g)
        elif self.rosta == 'RIM':
            d0_g_tt = self.decoder0_g_tt(paddle.concat((self.psp1(psp
                ), d1_g), 1))
            d0_g_ft = self.decoder0_g_ft(paddle.concat((self.psp1(psp
                ), d1_g), 1))
            d0_g_bt = self.decoder0_g_bt(paddle.concat((self.psp1(psp
                ), d1_g), 1))
        else:
            d0_g = self.decoder0_g(paddle.concat((self.psp1(psp),
                d1_g), 1))
        if self.rosta == 'RIM':
            glance_sigmoid_tt = F.sigmoid(d0_g_tt)
            glance_sigmoid_ft = F.sigmoid(d0_g_ft)
            glance_sigmoid_bt = F.sigmoid(d0_g_bt)
        else:
            glance_sigmoid = F.sigmoid(d0_g)
        if self.backbone == 'r34_2b':
            bb = self.bridge_block(e6)
            d6_f = self.decoder6_f(paddle.concat((bb, e6), 1))
            d5_f = self.decoder5_f(paddle.concat((d6_f, e5), 1))
            d4_f = self.decoder4_f(paddle.concat((d5_f, e4), 1))
        else:
            bb = self.bridge_block(e4)
            d4_f = self.decoder4_f(paddle.concat((bb, e4), 1))
        d3_f = self.decoder3_f(paddle.concat((d4_f, e3), 1))
        d2_f = self.decoder2_f(paddle.concat((d3_f, e2), 1))
        d1_f = self.decoder1_f(paddle.concat((d2_f, e1), 1))
        if self.backbone == 'r34_2b':
            if self.rosta == 'RIM':
                d0_f_tt = self.decoder0_f_tt(d1_f)
                d0_f_ft = self.decoder0_f_ft(d1_f)
                d0_f_bt = self.decoder0_f_bt(d1_f)
            else:
                d0_f = self.decoder0_f(d1_f)
        elif self.rosta == 'RIM':
            d0_f_tt = self.decoder0_f_tt(paddle.concat((d1_f, e0), 1))
            d0_f_ft = self.decoder0_f_ft(paddle.concat((d1_f, e0), 1))
            d0_f_bt = self.decoder0_f_bt(paddle.concat((d1_f, e0), 1))
        else:
            d0_f = self.decoder0_f(paddle.concat((d1_f, e0), 1))
        if self.rosta == 'RIM':
            focus_sigmoid_tt = F.sigmoid(d0_f_tt)
            focus_sigmoid_ft = F.sigmoid(d0_f_ft)
            focus_sigmoid_bt = F.sigmoid(d0_f_bt)
        else:
            focus_sigmoid = F.sigmoid(d0_f)
        if self.rosta == 'RIM':
            fusion_sigmoid_tt = collaborative_matting('TT',
                glance_sigmoid_tt, focus_sigmoid_tt)
            fusion_sigmoid_ft = collaborative_matting('FT',
                glance_sigmoid_ft, focus_sigmoid_ft)
            fusion_sigmoid_bt = collaborative_matting('BT',
                glance_sigmoid_bt, focus_sigmoid_bt)
            fusion_sigmoid = paddle.concat((fusion_sigmoid_tt,
                fusion_sigmoid_ft, fusion_sigmoid_bt), 1)
            fusion_sigmoid = self.rim(fusion_sigmoid)
            return [[glance_sigmoid_tt, focus_sigmoid_tt, fusion_sigmoid_tt
                ], [glance_sigmoid_ft, focus_sigmoid_ft, fusion_sigmoid_ft],
                [glance_sigmoid_bt, focus_sigmoid_bt, fusion_sigmoid_bt],
                fusion_sigmoid]
        else:
            fusion_sigmoid = collaborative_matting(self.rosta,
                glance_sigmoid, focus_sigmoid)
            return glance_sigmoid, focus_sigmoid, fusion_sigmoid
            

def collaborative_matting(rosta, glance_sigmoid, focus_sigmoid):
    if rosta == 'TT':
        values = paddle.max(glance_sigmoid, axis=1)
        index = paddle.argmax(glance_sigmoid, axis=1)
        index = index[:, None, :, :].float()
        bg_mask = index.clone()
        bg_mask[bg_mask == 2] = 1
        bg_mask = 1 - bg_mask
        trimap_mask = index.clone()
        trimap_mask[trimap_mask == 2] = 0
        fg_mask = index.clone()
        fg_mask[fg_mask == 1] = 0
        fg_mask[fg_mask == 2] = 1
        focus_sigmoid = focus_sigmoid.cpu()
        trimap_mask = trimap_mask.cpu()
        fg_mask = fg_mask.cpu()
        fusion_sigmoid = focus_sigmoid * trimap_mask + fg_mask
    elif rosta == 'BT':
        values = paddle.max(glance_sigmoid, axis=1)
        index = paddle.argmax(glance_sigmoid, axis=1)
        index = index[:, None, :, :].float()
        fusion_sigmoid = index - focus_sigmoid
        fusion_sigmoid[fusion_sigmoid < 0] = 0
    else:
        values = paddle.max(glance_sigmoid, axis=1)
        index = paddle.argmax(glance_sigmoid, axis=1)
        index = index[:, None, :, :].float()
        fusion_sigmoid = index + focus_sigmoid
        fusion_sigmoid[fusion_sigmoid > 1] = 1
    return fusion_sigmoid


if __name__ ==  "__main__":
    model = GFM()
    x = paddle.ones([1,3, 256,256])
    result = model(x)
    print(x)