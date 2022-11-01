# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def pad_new(x, pad, mode="constant", value=0):
    new_pad = []
    for _ in range(x.ndim * 2 - len(pad)):
        new_pad.append(0)
    ndim = list(range(x.ndim - 1, 0, -1))
    axes_start = {}
    for i, _pad in enumerate(pad):
        if _pad < 0:
            new_pad.append(0)
            zhengshu, yushu = divmod(i, 2)
            if yushu == 0:
                axes_start[ndim[zhengshu]] = -_pad
        else:
            new_pad.append(_pad)

    padded = paddle.nn.functional.pad(x, new_pad, mode=mode, value=value)
    padded_shape = paddle.shape(padded)
    axes = []
    starts = []
    ends = []
    for k, v in axes_start.items():
        axes.append(k)
        starts.append(v)
        ends.append(padded_shape[k])
        assert v < padded_shape[k]

    if axes:
        return padded.slice(axes=axes, starts=starts, ends=ends)
    else:
        return padded


class Upsample2D(nn.Layer):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.Conv2DTranspose(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2D(self.channels, self.out_channels, 3, padding=1)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                x = self.conv(x)
            else:
                x = self.Conv2d_0(x)

        return x


class Downsample2D(nn.Layer):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2D(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2D(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            x = pad_new(x, pad, mode="constant", value=0)

        assert x.shape[1] == self.channels
        x = self.conv(x)

        return x


class FirUpsample2D(nn.Layer):

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2D(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(self, x, w=None, k=None, factor=2, gain=1):
        """Fused `upsample_2d()` followed by `Conv2d()`.

        Args:
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of arbitrary:
        order.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
        w: Weight tensor of the shape `[filterH, filterW, inChannels,
            outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same datatype as
        `x`.
        """

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if k is None:
            k = [1] * factor

        # setup kernel
        k = np.asarray(k, dtype=np.float32)
        if k.ndim == 1:
            k = np.outer(k, k)
        k /= np.sum(k)

        k = k * (gain * (factor**2))

        if self.use_conv:
            convH = w.shape[2]
            convW = w.shape[3]
            inC = w.shape[1]

            p = (k.shape[0] - factor) - (convW - 1)

            stride = (factor, factor)
            # Determine data dimensions.
            stride = [1, 1, factor, factor]
            output_shape = ((x.shape[2] - 1) * factor + convH, (x.shape[3] - 1) * factor + convW)
            output_padding = (
                output_shape[0] - (x.shape[2] - 1) * stride[0] - convH,
                output_shape[1] - (x.shape[3] - 1) * stride[1] - convW,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            inC = w.shape[1]
            num_groups = x.shape[1] // inC

            # Transpose weights.
            w = paddle.reshape(w, (num_groups, -1, inC, convH, convW))
            w = w[..., ::-1, ::-1].transpose([0, 2, 1, 3, 4])
            w = paddle.reshape(w, (num_groups * inC, -1, convH, convW))

            x = F.conv2d_transpose(x, w, stride=stride, output_padding=output_padding, padding=0)

            x = upfirdn2d_native(x, paddle.to_tensor(k), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))
        else:
            p = k.shape[0] - factor
            x = upfirdn2d_native(x, paddle.to_tensor(k), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))

        return x

    def forward(self, x):
        if self.use_conv:
            h = self._upsample_2d(x, self.Conv2d_0.weight, k=self.fir_kernel)
            h = h + self.Conv2d_0.bias.reshape([1, -1, 1, 1])
        else:
            h = self._upsample_2d(x, k=self.fir_kernel, factor=2)

        return h


class FirDownsample2D(nn.Layer):

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2D(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(self, x, w=None, k=None, factor=2, gain=1):
        """Fused `Conv2d()` followed by `downsample_2d()`.

        Args:
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of arbitrary:
        order.
            x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`. w: Weight tensor of the shape `[filterH,
            filterW, inChannels, outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] //
            numGroups`. k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] *
            factor`, which corresponds to average pooling. factor: Integer downsampling factor (default: 2). gain:
            Scaling factor for signal magnitude (default: 1.0).

        Returns:
            Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same
            datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        if k is None:
            k = [1] * factor

        # setup kernel
        k = np.asarray(k, dtype=np.float32)
        if k.ndim == 1:
            k = np.outer(k, k)
        k /= np.sum(k)

        k = k * gain

        if self.use_conv:
            _, _, convH, convW = w.shape
            p = (k.shape[0] - factor) + (convW - 1)
            s = [factor, factor]
            x = upfirdn2d_native(x, paddle.to_tensor(k), pad=((p + 1) // 2, p // 2))
            x = F.conv2d(x, w, stride=s, padding=0)
        else:
            p = k.shape[0] - factor
            x = upfirdn2d_native(x, paddle.to_tensor(k), down=factor, pad=((p + 1) // 2, p // 2))

        return x

    def forward(self, x):
        if self.use_conv:
            x = self._downsample_2d(x, w=self.Conv2d_0.weight, k=self.fir_kernel)
            x = x + self.Conv2d_0.bias.reshape([1, -1, 1, 1])
        else:
            x = self._downsample_2d(x, k=self.fir_kernel, factor=2)

        return x


class ResnetBlock(nn.Layer):

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_nin_shortcut=None,
        up=False,
        down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, epsilon=eps)

        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, epsilon=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.Silu()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, k=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, k=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_nin_shortcut = self.in_channels != self.out_channels if use_nin_shortcut is None else use_nin_shortcut

        self.conv_shortcut = None
        if self.use_nin_shortcut:
            self.conv_shortcut = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb, hey=False):
        h = x

        # make sure hidden states is in float32
        # when running in half-precision
        h = self.norm1(h.astype("float32")).astype(h.dtype)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            h = h + temb

        # make sure hidden states is in float32
        # when running in half-precision
        h = self.norm2(h.astype("float32")).astype(h.dtype)
        h = self.nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = (x + h) / self.output_scale_factor

        return out


class Mish(nn.Layer):

    def forward(self, x):
        return x * F.tanh(F.softplus(x))


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""Upsample2D a batch of 2D images with the given filter.

    Args:
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is a:
    multiple of the upsampling factor.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor

    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)

    k = k * (gain * (factor**2))
    p = k.shape[0] - factor
    return upfirdn2d_native(x, paddle.to_tensor(k), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""Downsample2D a batch of 2D images with the given filter.

    Args:
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor

    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)

    k = k * gain
    p = k.shape[0] - factor
    return upfirdn2d_native(x, paddle.to_tensor(k), down=factor, pad=((p + 1) // 2, p // 2))


def upfirdn2d_native(input, kernel, up=1, down=1, pad=(0, 0)):
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = input.shape
    input = input.reshape([-1, in_h, in_w, 1])

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.reshape([-1, in_h, 1, in_w, 1, minor])
    # TODO
    out = pad_new(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.reshape([-1, in_h * up_y, in_w * up_x, minor])

    out = pad_new(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :, ]

    out = out.transpose([0, 3, 1, 2])
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = paddle.flip(kernel, [0, 1]).reshape([1, 1, kernel_h, kernel_w])
    out = F.conv2d(out, w)
    out = out.reshape(
        [-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1])
    out = out.transpose([0, 2, 3, 1])
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.reshape([-1, channel, out_h, out_w])
