#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import time
import sys
import paddle.fluid as fluid
import math


class TSN_ResNet():
    def __init__(self, layers=50, seg_num=7, is_training=True):
        self.layers = 101  #layers
        self.seg_num = seg_num
        self.is_training = is_training

    def conv_bn_layer(self, input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=(not self.is_training),
            param_attr=fluid.param_attr.ParamAttr(name=bn_name + "_scale"),
            bias_attr=fluid.param_attr.ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu', name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0, num_filters=num_filters, filter_size=3, stride=stride, act='relu', name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None, name=name + "_branch2c")

        short = self.shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

    def net(self, input, class_dim=101):
        layers = self.layers
        seg_num = self.seg_num
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        # reshape input
        channels = input.shape[2]
        short_size = input.shape[3]
        input = fluid.layers.reshape(x=input, shape=[-1, channels, short_size, short_size])

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name='conv1')
        conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)

                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    name=conv_name)

        pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)

        feature = fluid.layers.reshape(x=pool, shape=[-1, seg_num, pool.shape[1]])
        return feature
