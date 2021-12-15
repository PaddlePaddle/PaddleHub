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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import ginet_resnet101vd_ade20k.layers as L

class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 shortcut: bool = True,
                 if_first: bool = False,
                 name: str = None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = L.ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = L.ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")

        if not shortcut:
            self.short = L.ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.elementwise_add(x=short, y=conv1, act='relu')

        return y


class ResNet101_vd(nn.Layer):
    def __init__(self,
                 multi_grid: tuple = (1, 2, 4)):
        super(ResNet101_vd, self).__init__()
        depth = [3, 4, 23, 3]
        num_channels = [64, 256, 512, 1024] 
        num_filters = [64, 128, 256, 512]
        self.feat_channels = [c * 4 for c in num_filters]
        dilation_dict = {2: 2, 3: 4}
        self.conv1_1 = L.ConvBNLayer(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            name="conv1_1")
        self.conv1_2 = L.ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = L.ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.stage_list = []

        for block in range(len(depth)):
            shortcut = False
            block_list = []
            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)
                dilation_rate = dilation_dict[
                    block] if dilation_dict and block in dilation_dict else 1
                if block == 3:
                    dilation_rate = dilation_rate * multi_grid[i]
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    L.BottleneckBlock(
                        in_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        out_channels=num_filters[block],
                        stride=2 if i == 0 and block != 0
                                    and dilation_rate == 1 else 1,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        name=conv_name,
                        dilation=dilation_rate))
                block_list.append(bottleneck_block)
                shortcut = True
            self.stage_list.append(block_list)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        feat_list = []
        for stage in self.stage_list:
            for block in stage:
                y = block(y)
            feat_list.append(y)
        return feat_list