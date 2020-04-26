# coding=utf-8
from __future__ import absolute_import

import paddle.fluid as fluid

__all__ = ["face_landmark_localization"]


def face_landmark_localization(image):
    # image = fluid.layers.data(shape=[1, 60, 60], name='data', dtype='float32')
    Conv1 = fluid.layers.conv2d(
        image,
        param_attr='Conv1_weights',
        name='Conv1',
        dilation=[1, 1],
        filter_size=[5, 5],
        stride=[1, 1],
        groups=1,
        bias_attr='Conv1_bias',
        padding=[2, 2],
        num_filters=20)
    ActivationTangH1 = fluid.layers.tanh(Conv1, name='ActivationTangH1')
    ActivationAbs1 = fluid.layers.abs(ActivationTangH1, name='ActivationAbs1')
    Pool1 = fluid.layers.pool2d(
        ActivationAbs1,
        exclusive=False,
        pool_type='max',
        pool_padding=[0, 0],
        name='Pool1',
        global_pooling=False,
        pool_stride=[2, 2],
        ceil_mode=True,
        pool_size=[2, 2])
    Conv2 = fluid.layers.conv2d(
        Pool1,
        param_attr='Conv2_weights',
        name='Conv2',
        dilation=[1, 1],
        filter_size=[5, 5],
        stride=[1, 1],
        groups=1,
        bias_attr='Conv2_bias',
        padding=[2, 2],
        num_filters=48)
    ActivationTangH2 = fluid.layers.tanh(Conv2, name='ActivationTangH2')
    ActivationAbs2 = fluid.layers.abs(ActivationTangH2, name='ActivationAbs2')
    Pool2 = fluid.layers.pool2d(
        ActivationAbs2,
        exclusive=False,
        pool_type='max',
        pool_padding=[0, 0],
        name='Pool2',
        global_pooling=False,
        pool_stride=[2, 2],
        ceil_mode=True,
        pool_size=[2, 2])
    Conv3 = fluid.layers.conv2d(
        Pool2,
        param_attr='Conv3_weights',
        name='Conv3',
        dilation=[1, 1],
        filter_size=[3, 3],
        stride=[1, 1],
        groups=1,
        bias_attr='Conv3_bias',
        padding=[0, 0],
        num_filters=64)
    ActivationTangH3 = fluid.layers.tanh(Conv3, name='ActivationTangH3')
    ActivationAbs3 = fluid.layers.abs(ActivationTangH3, name='ActivationAbs3')
    Pool3 = fluid.layers.pool2d(
        ActivationAbs3,
        exclusive=False,
        pool_type='max',
        pool_padding=[0, 0],
        name='Pool3',
        global_pooling=False,
        pool_stride=[2, 2],
        ceil_mode=True,
        pool_size=[3, 3])
    Conv4 = fluid.layers.conv2d(
        Pool3,
        param_attr='Conv4_weights',
        name='Conv4',
        dilation=[1, 1],
        filter_size=[3, 3],
        stride=[1, 1],
        groups=1,
        bias_attr='Conv4_bias',
        padding=[0, 0],
        num_filters=80)
    ActivationTangH4 = fluid.layers.tanh(Conv4, name='ActivationTangH4')
    ActivationAbs4 = fluid.layers.abs(ActivationTangH4, name='ActivationAbs4')
    Dense1 = fluid.layers.fc(
        ActivationAbs4,
        param_attr='Dense1_weights',
        act=None,
        name='Dense1',
        size=512,
        bias_attr='Dense1_bias')
    ActivationTangH5 = fluid.layers.tanh(Dense1, name='ActivationTangH5')
    ActivationAbs5 = fluid.layers.abs(ActivationTangH5, name='ActivationAbs5')
    Dense3 = fluid.layers.fc(
        ActivationAbs5,
        param_attr='Dense3_weights',
        act=None,
        name='Dense3',
        size=136,
        bias_attr='Dense3_bias')
    return Dense3
