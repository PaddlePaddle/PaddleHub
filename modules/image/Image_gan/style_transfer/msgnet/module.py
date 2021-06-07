import os
import argparse

import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F

from paddlehub.env import MODULE_HOME
from paddlehub.module.module import moduleinfo
from paddlehub.vision.transforms import Compose, Resize, CenterCrop
from paddlehub.module.cv_module import StyleTransferModule


class GramMatrix(nn.Layer):
    """Calculate gram matrix"""

    def forward(self, y):
        (b, ch, h, w) = y.shape
        features = y.reshape((b, ch, w * h))
        features_t = features.transpose((0, 2, 1))
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class ConvLayer(nn.Layer):
    """Basic conv layer with reflection padding layer"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(ConvLayer, self).__init__()
        pad = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.Pad2D([pad, pad, pad, pad], mode='reflect')
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: paddle.Tensor):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Layer):
    """
    Upsamples the input and then does a convolution. This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/

    Args:
       in_channels(int): Number of input channels.
       out_channels(int): Number of output channels.
       kernel_size(int): Number of kernel size.
       stride(int): Number of stride.
       upsample(int): Scale factor for upsample layer, default is None.

    Return:
        img(paddle.Tensor): UpsampleConvLayer output.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        self.pad = int(np.floor(kernel_size / 2))
        if self.pad != 0:
            self.reflection_pad = nn.Pad2D([self.pad, self.pad, self.pad, self.pad], mode='reflect')
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.pad != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Bottleneck(nn.Layer):
    """ Pre-activation residual block
        Identity Mapping in Deep Residual Networks
        ref https://arxiv.org/abs/1603.05027

    Args:
       inplanes(int): Number of input channels.
       planes(int): Number of output channels.
       stride(int): Number of stride.
       downsample(int): Scale factor for downsample layer, default is None.
       norm_layer(nn.Layer): Batch norm layer, default is nn.BatchNorm2D.

    Return:
        img(paddle.Tensor): Bottleneck output.
    """

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: int = None,
                 norm_layer: nn.Layer = nn.BatchNorm2D):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2D(inplanes, planes * self.expansion, kernel_size=1, stride=stride)
        conv_block = (norm_layer(inplanes), nn.ReLU(), nn.Conv2D(inplanes, planes, kernel_size=1, stride=1),
                      norm_layer(planes), nn.ReLU(), ConvLayer(planes, planes, kernel_size=3, stride=stride),
                      norm_layer(planes), nn.ReLU(), nn.Conv2D(
                          planes, planes * self.expansion, kernel_size=1, stride=1))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: paddle.Tensor):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        m = self.conv_block(x)
        return residual + self.conv_block(x)


class UpBottleneck(nn.Layer):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953

    Args:
       inplanes(int): Number of input channels.
       planes(int): Number of output channels.
       stride(int): Number of stride, default is 2.
       norm_layer(nn.Layer): Batch norm layer, default is nn.BatchNorm2D.

    Return:
        img(paddle.Tensor): UpBottleneck output.
    """

    def __init__(self, inplanes: int, planes: int, stride: int = 2, norm_layer: nn.Layer = nn.BatchNorm2D):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(
            inplanes, planes * self.expansion, kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes), nn.ReLU(), nn.Conv2D(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [
            norm_layer(planes),
            nn.ReLU(),
            UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)
        ]
        conv_block += [
            norm_layer(planes),
            nn.ReLU(),
            nn.Conv2D(planes, planes * self.expansion, kernel_size=1, stride=1)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: paddle.Tensor):
        return self.residual_layer(x) + self.conv_block(x)


class Inspiration(nn.Layer):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953

    Args:
       C(int): Number of input channels.
       B(int):  B is equal to 1 or input mini_batch, default is 1.

    Return:
        img(paddle.Tensor): UpBottleneck output.
    """

    def __init__(self, C: int, B: int = 1):
        super(Inspiration, self).__init__()

        self.weight = self.weight = paddle.create_parameter(shape=[1, C, C], dtype='float32')
        # non-parameter buffer
        self.G = paddle.to_tensor(np.random.rand(B, C, C))
        self.C = C

    def setTarget(self, target: paddle.Tensor):
        self.G = target

    def forward(self, X: paddle.Tensor):
        # input X is a 3D feature map
        self.P = paddle.bmm(self.weight.expand_as(self.G), self.G)

        x = paddle.bmm(
            self.P.transpose((0, 2, 1)).expand((X.shape[0], self.C, self.C)), X.reshape((X.shape[0], X.shape[1],
                                                                                         -1))).reshape(X.shape)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x ' + str(self.C) + ')'


class Vgg16(nn.Layer):
    """ First four layers from Vgg16."""

    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

        checkpoint = os.path.join(MODULE_HOME, 'msgnet', 'vgg16.pdparams')
        model_dict = paddle.load(checkpoint)
        self.set_dict(model_dict)
        print("load pretrained vgg16 checkpoint success")

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]


@moduleinfo(
    name="msgnet",
    type="CV/image_editing",
    author="baidu-vis",
    author_email="",
    summary="Msgnet is a image colorization style transfer model, this module is trained with COCO2014 dataset.",
    version="1.0.0",
    meta=StyleTransferModule)
class MSGNet(nn.Layer):
    """ MSGNet (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953

    Args:
       input_nc(int): Number of input channels, default is 3.
       output_nc(int): Number of output channels, default is 3.
       ngf(int): Number of input channel for middle layer, default is 128.
       n_blocks(int): Block number, default is 6.
       norm_layer(nn.Layer): Batch norm layer, default is nn.InstanceNorm2D.
       load_checkpoint(str): Pretrained checkpoint path, default is None.

    Return:
        img(paddle.Tensor): MSGNet output.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=128, n_blocks=6, norm_layer=nn.InstanceNorm2D,
                 load_checkpoint=None):
        super(MSGNet, self).__init__()
        self.gram = GramMatrix()
        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = [
            ConvLayer(input_nc, 64, kernel_size=7, stride=1),
            norm_layer(64),
            nn.ReLU(),
            block(64, 32, 2, 1, norm_layer),
            block(32 * expansion, ngf, 2, 1, norm_layer)
        ]

        self.model1 = nn.Sequential(*tuple(model1))

        model = []
        model += model1

        self.ins = Inspiration(ngf * expansion)
        model.append(self.ins)
        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

        model += [
            upblock(ngf * expansion, 32, 2, norm_layer),
            upblock(32 * expansion, 16, 2, norm_layer),
            norm_layer(16 * expansion),
            nn.ReLU(),
            ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)
        ]
        model = tuple(model)
        self.model = nn.Sequential(*model)

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'style_paddle.pdparams')
            model_dict = paddle.load(checkpoint)
            model_dict_clone = model_dict.copy()
            for key, value in model_dict_clone.items():
                if key.endswith(("scale")):
                    name = key.rsplit('.', 1)[0] + '.bias'
                    model_dict[name] = paddle.zeros(shape=model_dict[name].shape, dtype='float32')
                    model_dict[key] = paddle.ones(shape=model_dict[key].shape, dtype='float32')
            self.set_dict(model_dict)
            self.model_dict = model_dict
            print("load pretrained checkpoint success")

        self._vgg = None

    def transform(self, path: str):
        transform = Compose([Resize((256, 256), interpolation='LINEAR')])
        return transform(path)

    def setTarget(self, Xs: paddle.Tensor):
        """Calculate feature gram matrix"""
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def getFeature(self, input: paddle.Tensor):
        if not self._vgg:
            self._vgg = Vgg16()
        return self._vgg(input)

    def forward(self, input: paddle.Tensor):
        return self.model(input)
