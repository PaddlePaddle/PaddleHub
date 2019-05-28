#coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageEnhance
import numpy as np

from paddlehub.common import utils


def _check_range_0_1(value):
    value = value if value <= 1 else 1
    value = value if value >= 0 else 0
    return value


def _check_bound(low, high):
    low = _check_range_0_1(low)
    high = _check_range_0_1(high)
    high = high if high >= low else low
    return low, high


def _check_img(img):
    if isinstance(img, str):
        utils.check_path(img)
        img = Image.open(img)
    return img


def _check_img_and_size(img, width, height):
    img = _check_img(img)
    img_width, img_height = img.size
    height = height if img_height > height else img_height
    height = img_height if height <= 0 else height
    width = width if img_width > width else img_width
    width = img_width if width <= 0 else width
    return img, width, height


def image_crop_from_position(img, width, height, w_start, h_start):
    img, width, height = _check_img_and_size(img, width, height)
    w_end = w_start + width
    h_end = h_start + height
    return img.crop((w_start, h_start, w_end, h_end))


def image_crop_from_TL(img, width, height):
    w_start = h_start = 0
    return image_crop_from_position(img, width, height, w_start, h_start)


def image_crop_from_TR(img, width, height):
    img, width, height = _check_img_and_size(img, width, height)
    w_start = img.size[0] - width
    h_start = 0
    return image_crop_from_position(img, width, height, w_start, h_start)


def image_crop_from_BL(img, width, height):
    img, width, height = _check_img_and_size(img, width, height)
    w_start = 0
    h_start = img.size[1] - height
    return image_crop_from_position(img, width, height, w_start, h_start)


def image_crop_from_BR(img, width, height):
    img, width, height = _check_img_and_size(img, width, height)
    w_start = img.size[0] - width
    h_start = img.size[1] - height
    return image_crop_from_position(img, width, height, w_start, h_start)


def image_crop_from_centor(img, width, height):
    img = _check_img(img)
    w_start = (img.size[0] - width) / 2
    h_start = (img.size[1] - height) / 2
    return image_crop_from_position(img, width, height, w_start, h_start)


def image_crop_random(img, width=0, height=0):
    img = _check_img(img)
    width = width if width else np.random.randint(
        int(img.size[0] / 10), img.size[0])
    height = height if height else np.random.randint(
        int(img.size[1] / 10), img.size[1])
    w_start = np.random.randint(0, img.size[0] - width)
    h_start = np.random.randint(0, img.size[1] - height)
    return image_crop_from_position(img, width, height, w_start, h_start)


def image_resize(img, width, height, interpolation_method=Image.LANCZOS):
    img = _check_img(img)
    return img.resize((width, height), interpolation_method)


def image_resize_random(img,
                        width=0,
                        height=0,
                        interpolation_method=Image.LANCZOS):
    img = _check_img(img)
    width = width if width else np.random.randint(
        int(img.size[0] / 10), img.size[0])
    height = height if height else np.random.randint(
        int(img.size[1] / 10), img.size[1])
    return image_resize(img, width, height, interpolation_method)


def image_rotate(img, angle, expand=False):
    img = _check_img(img)
    return img.rotate(angle, expand=expand)


def image_rotate_random(img, low=0, high=360, expand=False):
    angle = np.random.randint(low, high)
    return image_rotate(img, angle, expand)


def image_brightness_adjust(img, delta):
    delta = _check_range_0_1(delta)
    img = _check_img(img)
    return ImageEnhance.Brightness(img).enhance(delta)


def image_brightness_adjust_random(img, low=0, high=1):
    low, high = _check_bound(low, high)
    delta = np.random.uniform(low, high)
    return image_brightness_adjust(img, delta)


def image_contrast_adjust(img, delta):
    delta = _check_range_0_1(delta)
    img = _check_img(img)
    return ImageEnhance.Contrast(img).enhance(delta)


def image_contrast_adjust_random(img, low=0, high=1):
    low, high = _check_bound(low, high)
    delta = np.random.uniform(low, high)
    return image_contrast_adjust(img, delta)


def image_saturation_adjust(img, delta):
    delta = _check_range_0_1(delta)
    img = _check_img(img)
    return ImageEnhance.Color(img).enhance(delta)


def image_saturation_adjust_random(img, low=0, high=1):
    low, high = _check_bound(low, high)
    delta = np.random.uniform(low, high)
    return image_saturation_adjust(img, delta)


def image_flip_top_bottom(img):
    img = _check_img(img)
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def image_flip_left_right(img):
    img = _check_img(img)
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def image_flip_random(img):
    img = _check_img(img)
    flag = np.random.randint(0, 1)
    if flag:
        return image_flip_top_bottom(img)
    else:
        return image_flip_left_right(img)


def image_random_process(img,
                         enable_resize=True,
                         enable_crop=True,
                         enable_rotate=True,
                         enable_brightness_adjust=True,
                         enable_contrast_adjust=True,
                         enable_saturation_adjust=True,
                         enable_flip=True):
    operator_list = []
    if enable_resize:
        operator_list.append(image_resize_random)
    if enable_crop:
        operator_list.append(image_crop_random)
    if enable_rotate:
        operator_list.append(image_rotate_random)
    if enable_brightness_adjust:
        operator_list.append(image_brightness_adjust_random)
    if enable_contrast_adjust:
        operator_list.append(image_contrast_adjust_random)
    if enable_saturation_adjust:
        operator_list.append(image_saturation_adjust_random)
    if enable_flip:
        operator_list.append(image_flip_random)

    if not operator_list:
        return img

    random_op_index = np.random.randint(0, len(operator_list) - 1)
    random_op = operator_list[random_op_index]
    return random_op(img)
