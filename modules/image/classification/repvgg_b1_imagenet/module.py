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
from typing import Union

import paddle
import paddle.nn as nn
import paddlehub.vision.transforms as T
import numpy as np
from paddlehub.module.module import moduleinfo
from paddlehub.module.cv_module import ImageClassifierModule


class ConvBN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(num_features=out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class RepVGGBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros'):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = nn.BatchNorm2D(
            num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
        self.rbr_1x1 = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups)

    def forward(self, inputs):
        if not self.training:
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def eval(self):
        if not hasattr(self, 'rbr_reparam'):
            self.rbr_reparam = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                padding_mode=self.padding_mode)
        self.training = False
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        for layer in self.sublayers():
            layer.eval()

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


@moduleinfo(
    name="repvgg_b1_imagenet",
    type="CV/classification",
    author="paddlepaddle",
    author_email="",
    summary="repvgg_b1_imagenet is a classification model, "
    "this module is trained with Imagenet dataset.",
    version="1.0.0",
    meta=ImageClassifierModule)
class RepVGG_B1(nn.Layer):
    def __init__(self, label_list: list = None, load_checkpoint: str = None):
        super(RepVGG_B1, self).__init__()

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

        num_blocks = [4, 6, 16, 1]
        width_multiplier = [2, 2, 2, 4]
        self.override_groups_map = dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2D(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), class_dim)

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")
        else:
            checkpoint = os.path.join(self.directory, 'model.pdparams')
            self.model_dict = paddle.load(checkpoint)
            self.set_dict(self.model_dict)
            print("load pretrained checkpoint success")

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def transforms(self, images: Union[str, np.ndarray]):
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ],
                               to_rgb=True)
        return transforms(images).astype('float32')

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        feature = self.gap(out)
        out = paddle.flatten(feature, start_axis=1)
        out = self.linear(out)
        return out, feature
