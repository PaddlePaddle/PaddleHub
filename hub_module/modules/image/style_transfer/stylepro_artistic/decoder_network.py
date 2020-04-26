# coding=utf-8
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid


def decoder_net():
    x2paddle_22 = fluid.layers.create_parameter(
        dtype='float32',
        shape=[4],
        name='x2paddle_22',
        attr='x2paddle_22',
        default_initializer=Constant(0.0))
    x2paddle_36 = fluid.layers.create_parameter(
        dtype='float32',
        shape=[4],
        name='x2paddle_36',
        attr='x2paddle_36',
        default_initializer=Constant(0.0))
    x2paddle_44 = fluid.layers.create_parameter(
        dtype='float32',
        shape=[4],
        name='x2paddle_44',
        attr='x2paddle_44',
        default_initializer=Constant(0.0))
    x2paddle_input_1 = fluid.layers.data(
        dtype='float32',
        shape=[1, 512, 64, 64],
        name='x2paddle_input_1',
        append_batch_size=False)
    x2paddle_19 = fluid.layers.pad2d(
        x2paddle_input_1,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_19')
    x2paddle_20 = fluid.layers.conv2d(
        x2paddle_19,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_1',
        name='x2paddle_20',
        bias_attr='x2paddle_2')
    x2paddle_21 = fluid.layers.relu(x2paddle_20, name='x2paddle_21')
    x2paddle_23 = fluid.layers.resize_nearest(
        x2paddle_21, name='x2paddle_23', out_shape=[128, 128])
    x2paddle_24 = fluid.layers.pad2d(
        x2paddle_23,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_24')
    x2paddle_25 = fluid.layers.conv2d(
        x2paddle_24,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_3',
        name='x2paddle_25',
        bias_attr='x2paddle_4')
    x2paddle_26 = fluid.layers.relu(x2paddle_25, name='x2paddle_26')
    x2paddle_27 = fluid.layers.pad2d(
        x2paddle_26,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_27')
    x2paddle_28 = fluid.layers.conv2d(
        x2paddle_27,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_5',
        name='x2paddle_28',
        bias_attr='x2paddle_6')
    x2paddle_29 = fluid.layers.relu(x2paddle_28, name='x2paddle_29')
    x2paddle_30 = fluid.layers.pad2d(
        x2paddle_29,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_30')
    x2paddle_31 = fluid.layers.conv2d(
        x2paddle_30,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_7',
        name='x2paddle_31',
        bias_attr='x2paddle_8')
    x2paddle_32 = fluid.layers.relu(x2paddle_31, name='x2paddle_32')
    x2paddle_33 = fluid.layers.pad2d(
        x2paddle_32,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_33')
    x2paddle_34 = fluid.layers.conv2d(
        x2paddle_33,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_9',
        name='x2paddle_34',
        bias_attr='x2paddle_10')
    x2paddle_35 = fluid.layers.relu(x2paddle_34, name='x2paddle_35')
    x2paddle_37 = fluid.layers.resize_nearest(
        x2paddle_35, name='x2paddle_37', out_shape=[256, 256])
    x2paddle_38 = fluid.layers.pad2d(
        x2paddle_37,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_38')
    x2paddle_39 = fluid.layers.conv2d(
        x2paddle_38,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_11',
        name='x2paddle_39',
        bias_attr='x2paddle_12')
    x2paddle_40 = fluid.layers.relu(x2paddle_39, name='x2paddle_40')
    x2paddle_41 = fluid.layers.pad2d(
        x2paddle_40,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_41')
    x2paddle_42 = fluid.layers.conv2d(
        x2paddle_41,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_13',
        name='x2paddle_42',
        bias_attr='x2paddle_14')
    x2paddle_43 = fluid.layers.relu(x2paddle_42, name='x2paddle_43')
    x2paddle_45 = fluid.layers.resize_nearest(
        x2paddle_43, name='x2paddle_45', out_shape=[512, 512])
    x2paddle_46 = fluid.layers.pad2d(
        x2paddle_45,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_46')
    x2paddle_47 = fluid.layers.conv2d(
        x2paddle_46,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_15',
        name='x2paddle_47',
        bias_attr='x2paddle_16')
    x2paddle_48 = fluid.layers.relu(x2paddle_47, name='x2paddle_48')
    x2paddle_49 = fluid.layers.pad2d(
        x2paddle_48,
        pad_value=0.0,
        mode='reflect',
        paddings=[1, 1, 1, 1],
        name='x2paddle_49')
    x2paddle_50 = fluid.layers.conv2d(
        x2paddle_49,
        num_filters=3,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='x2paddle_17',
        name='x2paddle_50',
        bias_attr='x2paddle_18')
    return x2paddle_input_1, x2paddle_50
