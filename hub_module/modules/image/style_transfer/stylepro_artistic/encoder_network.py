# coding=utf-8
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid


def encoder_net():
    x2paddle_0 = fluid.layers.data(
        dtype='float32',
        shape=[1, 3, 512, 512],
        name='x2paddle_0',
        append_batch_size=False)
    x2paddle_21 = fluid.layers.conv2d(
        x2paddle_0,
        num_filters=3,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_1',
        name='x2paddle_21',
        bias_attr='x2paddle_2')
    x2paddle_22 = fluid.layers.pad2d(
        x2paddle_21,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_22')
    x2paddle_23 = fluid.layers.conv2d(
        x2paddle_22,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_3',
        name='x2paddle_23',
        bias_attr='x2paddle_4')
    x2paddle_24 = fluid.layers.relu(x2paddle_23, name='x2paddle_24')
    x2paddle_25 = fluid.layers.pad2d(
        x2paddle_24,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_25')
    x2paddle_26 = fluid.layers.conv2d(
        x2paddle_25,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_5',
        name='x2paddle_26',
        bias_attr='x2paddle_6')
    x2paddle_27 = fluid.layers.relu(x2paddle_26, name='x2paddle_27')
    x2paddle_28 = fluid.layers.pool2d(
        x2paddle_27,
        pool_size=[2, 2],
        pool_type='max',
        pool_stride=[2, 2],
        pool_padding=[0, 0],
        ceil_mode=False,
        name='x2paddle_28',
        exclusive=False)
    x2paddle_29 = fluid.layers.pad2d(
        x2paddle_28,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_29')
    x2paddle_30 = fluid.layers.conv2d(
        x2paddle_29,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_7',
        name='x2paddle_30',
        bias_attr='x2paddle_8')
    x2paddle_31 = fluid.layers.relu(x2paddle_30, name='x2paddle_31')
    x2paddle_32 = fluid.layers.pad2d(
        x2paddle_31,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_32')
    x2paddle_33 = fluid.layers.conv2d(
        x2paddle_32,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_9',
        name='x2paddle_33',
        bias_attr='x2paddle_10')
    x2paddle_34 = fluid.layers.relu(x2paddle_33, name='x2paddle_34')
    x2paddle_35 = fluid.layers.pool2d(
        x2paddle_34,
        pool_size=[2, 2],
        pool_type='max',
        pool_stride=[2, 2],
        pool_padding=[0, 0],
        ceil_mode=False,
        name='x2paddle_35',
        exclusive=False)
    x2paddle_36 = fluid.layers.pad2d(
        x2paddle_35,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_36')
    x2paddle_37 = fluid.layers.conv2d(
        x2paddle_36,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_11',
        name='x2paddle_37',
        bias_attr='x2paddle_12')
    x2paddle_38 = fluid.layers.relu(x2paddle_37, name='x2paddle_38')
    x2paddle_39 = fluid.layers.pad2d(
        x2paddle_38,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_39')
    x2paddle_40 = fluid.layers.conv2d(
        x2paddle_39,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_13',
        name='x2paddle_40',
        bias_attr='x2paddle_14')
    x2paddle_41 = fluid.layers.relu(x2paddle_40, name='x2paddle_41')
    x2paddle_42 = fluid.layers.pad2d(
        x2paddle_41,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_42')
    x2paddle_43 = fluid.layers.conv2d(
        x2paddle_42,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_15',
        name='x2paddle_43',
        bias_attr='x2paddle_16')
    x2paddle_44 = fluid.layers.relu(x2paddle_43, name='x2paddle_44')
    x2paddle_45 = fluid.layers.pad2d(
        x2paddle_44,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_45')
    x2paddle_46 = fluid.layers.conv2d(
        x2paddle_45,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_17',
        name='x2paddle_46',
        bias_attr='x2paddle_18')
    x2paddle_47 = fluid.layers.relu(x2paddle_46, name='x2paddle_47')
    x2paddle_48 = fluid.layers.pool2d(
        x2paddle_47,
        pool_size=[2, 2],
        pool_type='max',
        pool_stride=[2, 2],
        pool_padding=[0, 0],
        ceil_mode=False,
        name='x2paddle_48',
        exclusive=False)
    x2paddle_49 = fluid.layers.pad2d(
        x2paddle_48,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_49')
    x2paddle_50 = fluid.layers.conv2d(
        x2paddle_49,
        num_filters=512,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_19',
        name='x2paddle_50',
        bias_attr='x2paddle_20')
    x2paddle_51 = fluid.layers.relu(x2paddle_50, name='x2paddle_51')
    return x2paddle_0, x2paddle_51
