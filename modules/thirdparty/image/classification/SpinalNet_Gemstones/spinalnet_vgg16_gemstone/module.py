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

import paddle
from paddle import nn


class VGG(nn.Layer):
    def __init__(self, features, with_pool=True):
        super(VGG, self).__init__()
        self.features = features
        self.with_pool = with_pool

        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((7, 7))

    def forward(self, x):
        x = self.features(x)

        if self.with_pool:
            x = self.avgpool(x)

        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg16(batch_norm=False, **kwargs):
    model_name = 'vgg16'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'D', batch_norm, **kwargs)


@moduleinfo(
    name="spinalnet_vgg16_gemstone",
    type="CV/classification",
    author="nanting03",
    author_email="975348977@qq.com",
    summary="spinalnet_vgg16_gemstone is a classification model, "
    "this module is trained with Gemstone dataset.",
    version="1.0.0",
    meta=ImageClassifierModule)
class SpinalNet_VGG16(nn.Layer):
    def __init__(self, label_list: list = None, load_checkpoint: str = None):
        super(SpinalNet_VGG16, self).__init__()

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

        self.backbone = vgg16()

        half_in_size = round(512 * 7 * 7 / 2)
        layer_width = 4096

        self.half_in_size = half_in_size

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1D(layer_width),
            nn.ReLU(),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1D(layer_width),
            nn.ReLU(),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1D(layer_width),
            nn.ReLU(),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1D(layer_width),
            nn.ReLU(),
        )
        self.fc_out = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(layer_width * 4, class_dim))

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'spinalnet_vgg16.pdparams')
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
