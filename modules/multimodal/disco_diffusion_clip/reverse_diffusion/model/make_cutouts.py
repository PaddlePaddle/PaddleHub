import math

import discoart.model.transforms as T
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from resize_right import resize

skip_augs = False  # @param{type: 'boolean'}


def sinc(x):
    return paddle.where(x != 0, paddle.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = paddle.logical_and(-a < x, x < a)
    out = paddle.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = paddle.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return paddle.concat([-out[1:].flip([0]), out])[1:-1]


class MakeCutouts(nn.Layer):

    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = nn.Sequential(*[
            T.RandomHorizontalFlip(prob=0.5),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size *
                           paddle.zeros(1, ).normal_(mean=0.8, std=0.3).clip(float(self.cut_size / max_size), 1.0))
                offsetx = paddle.randint(0, abs(sideX - size + 1), ())
                offsety = paddle.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = paddle.concat(cutouts, axis=0)
        return cutouts


class MakeCutoutsDango(nn.Layer):

    def __init__(self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = nn.Sequential(*[
            T.RandomHorizontalFlip(prob=0.5),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(cutout[:, :, :, ::-1])
                if self.Overview == 4:
                    cutouts.append(gray(cutout[:, :, :, ::-1]))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(paddle.rand([1])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = paddle.randint(0, sideX - size + 1)
                offsety = paddle.randint(0, sideY - size + 1)
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = paddle.concat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


padargs = {}
