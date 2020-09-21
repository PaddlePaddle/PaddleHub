import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal


class ConvBNLayer(nn.Layer):
    """Basic block for Darknet"""
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 filter_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = 0,
                 act: str = 'leakly',
                 is_test: bool = False):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(ch_in,
                              ch_out,
                              filter_size,
                              padding=padding,
                              stride=stride,
                              groups=groups,
                              weight_attr=paddle.ParamAttr(initializer=Normal(0., 0.02)),
                              bias_attr=False)

        self.batch_norm = nn.BatchNorm(num_channels=ch_out,
                                       is_test=is_test,
                                       param_attr=paddle.ParamAttr(initializer=Normal(0., 0.02),
                                                                   regularizer=L2Decay(0.)))
        self.act = act

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == "leakly":
            out = F.leaky_relu(x=out, negative_slope=0.1)
        return out


class DownSample(nn.Layer):
    """Downsample block for Darknet"""
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 filter_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 is_test: bool = False):
        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in,
                                         ch_out=ch_out,
                                         filter_size=filter_size,
                                         stride=stride,
                                         padding=padding,
                                         is_test=is_test)
        self.ch_out = ch_out

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(nn.Layer):
    """Basic residual block for Darknet"""
    def __init__(self, ch_in: int, ch_out: int, is_test: bool = False):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, filter_size=1, stride=1, padding=0, is_test=is_test)
        self.conv2 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1, is_test=is_test)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.elementwise_add(x=inputs, y=conv2, act=None)
        return out


class LayerWarp(nn.Layer):
    """Warp layer composed by basic residual blocks"""
    def __init__(self, ch_in: int, ch_out: int, count: int, is_test: bool = False):
        super(LayerWarp, self).__init__()
        self.basicblock0 = BasicBlock(ch_in, ch_out, is_test=is_test)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i), BasicBlock(ch_out * 2, ch_out, is_test=is_test))
            self.res_out_list.append(res_out)
        self.ch_out = ch_out

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(nn.Layer):
    """Darknet53
    Args:
        ch_in(int): Input channels, default is 3.
        is_test (bool): Set the test mode, default is True.
    """
    def __init__(self, ch_in: int = 3, is_test: bool = False):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=32, filter_size=3, stride=1, padding=1, is_test=is_test)

        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2, is_test=is_test)
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]

        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer("stage_%d" % (i),
                                           LayerWarp(int(ch_in[i]), 32 * (2**i), stage, is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)

        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i, DownSample(ch_in=32 * (2**(i + 1)),
                                                      ch_out=32 * (2**(i + 2)),
                                                      is_test=is_test))
            self.downsample_list.append(downsample)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]
