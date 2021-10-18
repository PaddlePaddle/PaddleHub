# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import math

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

__all__ = ['DarkNet']


class DarkNet(object):
    """DarkNet, see https://pjreddie.com/darknet/yolo/
    Args:
        depth (int): network depth, currently only darknet 53 is supported
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported
        norm_decay (float): weight decay for normalization layer weights
        get_prediction (bool): whether to get prediction
        class_dim (int): number of class while classification
    """

    def __init__(self,
                 depth=53,
                 norm_type='sync_bn',
                 norm_decay=0.,
                 weight_prefix_name='',
                 get_prediction=False,
                 class_dim=1000):
        assert depth in [53], "unsupported depth value"
        self.depth = depth
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.depth_cfg = {53: ([1, 2, 8, 8, 4], self.basicblock)}
        self.prefix_name = weight_prefix_name
        self.class_dim = class_dim
        self.get_prediction = get_prediction

    def _conv_norm(self,
                   input,
                   ch_out,
                   filter_size,
                   stride,
                   padding,
                   act='leaky',
                   name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(float(self.norm_decay)),
            name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(float(self.norm_decay)),
            name=bn_name + '.offset')

        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        # leaky relu here has `alpha` as 0.1, can not be set by
        # `act` param in fluid.layers.batch_norm above.
        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)

        return out

    def _downsample(self,
                    input,
                    ch_out,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    name=None):
        return self._conv_norm(
            input,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            name=name)

    def basicblock(self, input, ch_out, name=None):
        conv1 = self._conv_norm(
            input,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            name=name + ".0")
        conv2 = self._conv_norm(
            conv1,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            name=name + ".1")
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self, block_func, input, ch_out, count, name=None):
        out = block_func(input, ch_out=ch_out, name='{}.0'.format(name))
        for j in six.moves.xrange(1, count):
            out = block_func(out, ch_out=ch_out, name='{}.{}'.format(name, j))
        return out

    def __call__(self, input):
        """Get the backbone of DarkNet, that is output for the 5 stages.

        :param input: Variable of input image
        :type input: Variable
        :Returns: The last variables of each stage.
        """
        stages, block_func = self.depth_cfg[self.depth]
        stages = stages[0:5]
        conv = self._conv_norm(
            input=input,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            name=self.prefix_name + "yolo_input")
        downsample_ = self._downsample(
            input=conv,
            ch_out=conv.shape[1] * 2,
            name=self.prefix_name + "yolo_input.downsample")
        blocks = []
        for i, stage in enumerate(stages):
            block = self.layer_warp(
                block_func=block_func,
                input=downsample_,
                ch_out=32 * 2**i,
                count=stage,
                name=self.prefix_name + "stage.{}".format(i))
            blocks.append(block)
            if i < len(stages) - 1:  # do not downsaple in the last stage
                downsample_ = self._downsample(
                    input=block,
                    ch_out=block.shape[1] * 2,
                    name=self.prefix_name + "stage.{}.downsample".format(i))
        if self.get_prediction:
            pool = fluid.layers.pool2d(
                input=block, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=self.class_dim,
                param_attr=ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv),
                    name='fc_weights'),
                bias_attr=ParamAttr(name='fc_offset'))
            out = fluid.layers.softmax(out)
            return out
        else:
            return blocks
