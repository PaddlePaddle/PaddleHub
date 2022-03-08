# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import os
import urllib.request

import cv2 as cv
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import functional
from PIL import Image
from ppgan.models.generators import DecoderNet
from ppgan.models.generators import Encoder
from ppgan.models.generators import RevisionNet
from ppgan.utils.visual import tensor2img


def img(img):
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    # HWC to CHW
    return img


def img_totensor(content_img, style_img):
    if content_img.ndim == 2:
        content_img = cv.cvtColor(content_img, cv.COLOR_GRAY2RGB)
    else:
        content_img = cv.cvtColor(content_img, cv.COLOR_BGR2RGB)
    h, w, c = content_img.shape
    content_img = Image.fromarray(content_img)
    content_img = content_img.resize((512, 512), Image.BILINEAR)
    content_img = np.array(content_img)
    content_img = img(content_img)
    content_img = functional.to_tensor(content_img)

    style_img = cv.cvtColor(style_img, cv.COLOR_BGR2RGB)
    style_img = Image.fromarray(style_img)
    style_img = style_img.resize((512, 512), Image.BILINEAR)
    style_img = np.array(style_img)
    style_img = img(style_img)
    style_img = functional.to_tensor(style_img)

    content_img = paddle.unsqueeze(content_img, axis=0)
    style_img = paddle.unsqueeze(style_img, axis=0)
    return content_img, style_img, h, w


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):
    """
    Laplacian

    return:
       x - upsample(downsample(x))
    """
    return x - tensor_resample(tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]), [x.shape[2], x.shape[3]])


def make_laplace_pyramid(x, levels):
    """
    Make Laplacian Pyramid
    """
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(current, (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid


def fold_laplace_pyramid(pyramid):
    """
    Fold Laplacian Pyramid
    """
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    return current


class LapStylePredictor:
    def __init__(self, weight_path=None):

        self.net_enc = Encoder()
        self.net_dec = DecoderNet()
        self.net_rev = RevisionNet()
        self.net_rev_2 = RevisionNet()

        self.net_enc.set_dict(paddle.load(weight_path)['net_enc'])
        self.net_enc.eval()
        self.net_dec.set_dict(paddle.load(weight_path)['net_dec'])
        self.net_dec.eval()
        self.net_rev.set_dict(paddle.load(weight_path)['net_rev'])
        self.net_rev.eval()
        self.net_rev_2.set_dict(paddle.load(weight_path)['net_rev_2'])
        self.net_rev_2.eval()

    def run(self, content_img, style_image):
        content_img, style_img, h, w = img_totensor(content_img, style_image)
        pyr_ci = make_laplace_pyramid(content_img, 2)
        pyr_si = make_laplace_pyramid(style_img, 2)
        pyr_ci.append(content_img)
        pyr_si.append(style_img)
        cF = self.net_enc(pyr_ci[2])
        sF = self.net_enc(pyr_si[2])
        stylized_small = self.net_dec(cF, sF)
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[pyr_ci[1], stylized_up], axis=1)
        stylized_rev_lap = self.net_rev(revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])

        stylized_up = F.interpolate(stylized_rev, scale_factor=2)

        revnet_input = paddle.concat(x=[pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap_second = self.net_rev_2(revnet_input)
        stylized_rev_second = fold_laplace_pyramid([stylized_rev_lap_second, stylized_rev_lap, stylized_small])

        stylized = stylized_rev_second
        stylized_visual = tensor2img(stylized, min_max=(0., 1.))

        return stylized_visual
