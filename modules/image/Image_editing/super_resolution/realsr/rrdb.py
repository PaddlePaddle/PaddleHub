import functools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.
    To create a registry (inside segmentron):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))

        return ret


class ResidualDenseBlock_5C(nn.Layer):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2D(nf, gc, 3, 1, 1, bias_attr=bias)
        self.conv2 = nn.Conv2D(nf + gc, gc, 3, 1, 1, bias_attr=bias)
        self.conv3 = nn.Conv2D(nf + 2 * gc, gc, 3, 1, 1, bias_attr=bias)
        self.conv4 = nn.Conv2D(nf + 3 * gc, gc, 3, 1, 1, bias_attr=bias)
        self.conv5 = nn.Conv2D(nf + 4 * gc, nf, 3, 1, 1, bias_attr=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(paddle.concat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(paddle.concat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(paddle.concat((x, x1, x2, x3), 1)))
        x5 = self.conv5(paddle.concat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Layer):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

GENERATORS = Registry("GENERATOR")

@GENERATORS.register()
class RRDBNet(nn.Layer):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        #### upsampling
        self.upconv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.upconv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.HRconv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(
            self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
