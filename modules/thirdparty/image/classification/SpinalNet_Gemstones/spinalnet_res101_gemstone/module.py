# copyright (c) 2021 nanting03. All Rights Reserve.
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
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
import paddlehub.vision.transforms as T
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


class BottleneckBlock(nn.Layer):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width, width, 3, padding=dilation, stride=stride, groups=groups, dilation=dilation, bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    def __init__(self, block=BottleneckBlock, depth=101, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        layers = layer_cfg[depth]
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, 1, stride=stride, bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        return x


@moduleinfo(
    name="spinalnet_res101_gemstone",
    type="CV/classification",
    author="nanting03",
    author_email="975348977@qq.com",
    summary="spinalnet_res101_gemstone is a classification model, "
    "this module is trained with Gemstone dataset.",
    version="1.0.0",
    meta=ImageClassifierModule)
class SpinalNet_ResNet101(nn.Layer):
    def __init__(self, label_list: list = None, load_checkpoint: str = None):
        super(SpinalNet_ResNet101, self).__init__()

        if label_list is not None:
            self.labels = label_list
            class_dim = len(self.labels)
        else:
            label_list = []
            label_file = os.path.join(self.directory, 'label_list.txt')
            files = open(label_file)
            for line in files.readlines():
                line = line.strip('\n')
                label_list.append(line)
            self.labels = label_list
            class_dim = len(self.labels)

        self.backbone = ResNet()

        half_in_size = round(2048 / 2)
        layer_width = 20

        self.half_in_size = half_in_size

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(half_in_size, layer_width), nn.BatchNorm1D(layer_width), nn.ReLU())
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(half_in_size + layer_width, layer_width), nn.BatchNorm1D(layer_width),
            nn.ReLU())
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(half_in_size + layer_width, layer_width), nn.BatchNorm1D(layer_width),
            nn.ReLU())
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(half_in_size + layer_width, layer_width), nn.BatchNorm1D(layer_width),
            nn.ReLU())
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(layer_width * 4, class_dim),
        )

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'spinalnet_res101.pdparams')
            self.model_dict = paddle.load(checkpoint)
            self.set_dict(self.model_dict)
            print("load pretrained checkpoint success")

    def transforms(self, images: Union[str, np.ndarray]):
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ],
                               to_rgb=True)
        return transforms(images)

    def forward(self, inputs: paddle.Tensor):
        y = self.backbone(inputs)
        feature = y
        y = paddle.flatten(y, 1)

        y1 = self.fc_spinal_layer1(y[:, 0:self.half_in_size])
        y2 = self.fc_spinal_layer2(paddle.concat([y[:, self.half_in_size:2 * self.half_in_size], y1], axis=1))
        y3 = self.fc_spinal_layer3(paddle.concat([y[:, 0:self.half_in_size], y2], axis=1))
        y4 = self.fc_spinal_layer4(paddle.concat([y[:, self.half_in_size:2 * self.half_in_size], y3], axis=1))

        y = paddle.concat([y1, y2, y3, y4], axis=1)

        y = self.fc_out(y)
        return y, feature
