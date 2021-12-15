# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import base64
from typing import Callable, Union, List, Tuple

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddleseg.transforms import functional
from PIL import Image


class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    """

    def __init__(self, transforms: Callable, to_rgb: bool = True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, data: dict) -> dict:

        if 'trans_info' not in data:
            data['trans_info'] = []
        for op in self.transforms:
            data = op(data)
            if data is None:
                return None

        data['img'] = np.transpose(data['img'], (2, 0, 1))
        for key in data.get('gt_fields', []):
            if len(data[key].shape) == 2:
                continue
            data[key] = np.transpose(data[key], (2, 0, 1))

        return data


class LoadImages:
    """
    Read images from image path.

    Args:
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.
    """
    def __init__(self, to_rgb: bool = True):
        self.to_rgb = to_rgb

    def __call__(self, data: dict) -> dict:

        if isinstance(data['img'], str):
            data['img'] = cv2.imread(data['img'])

        for key in data.get('gt_fields', []):
            if isinstance(data[key], str):
                data[key] = cv2.imread(data[key], cv2.IMREAD_UNCHANGED)
            # if alpha and trimap has 3 channels, extract one.
            if key in ['alpha', 'trimap']:
                if len(data[key].shape) > 2:
                    data[key] = data[key][:, :, 0]

        if self.to_rgb:
            data['img'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
            for key in data.get('gt_fields', []):
                if len(data[key].shape) == 2:
                    continue
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2RGB)

        return data


class LimitLong:
    """
    Limit the long edge of image.

    If the long edge is larger than max_long, resize the long edge
    to max_long, while scale the short edge proportionally.

    If the long edge is smaller than min_long, resize the long edge
    to min_long, while scale the short edge proportionally.

    Args:
        max_long (int, optional): If the long edge of image is larger than max_long,
            it will be resize to max_long. Default: None.
        min_long (int, optional): If the long edge of image is smaller than min_long,
            it will be resize to min_long. Default: None.
    """

    def __init__(self, max_long=None, min_long=None):
        if max_long is not None:
            if not isinstance(max_long, int):
                raise TypeError(
                    "Type of `max_long` is invalid. It should be int, but it is {}"
                    .format(type(max_long)))
        if min_long is not None:
            if not isinstance(min_long, int):
                raise TypeError(
                    "Type of `min_long` is invalid. It should be int, but it is {}"
                    .format(type(min_long)))
        if (max_long is not None) and (min_long is not None):
            if min_long > max_long:
                raise ValueError(
                    '`max_long should not smaller than min_long, but they are {} and {}'
                    .format(max_long, min_long))
        self.max_long = max_long
        self.min_long = min_long

    def __call__(self, data):
        h, w = data['img'].shape[:2]
        long_edge = max(h, w)
        target = long_edge
        if (self.max_long is not None) and (long_edge > self.max_long):
            target = self.max_long
        elif (self.min_long is not None) and (long_edge < self.min_long):
            target = self.min_long

        if target != long_edge:
            data['trans_info'].append(('resize', data['img'].shape[0:2]))
            data['img'] = functional.resize_long(data['img'], target)
            for key in data.get('gt_fields', []):
                data[key] = functional.resize_long(data[key], target)

        return data


class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5), std: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, data: dict) -> dict:
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        data['img'] = functional.normalize(data['img'], mean, std)
        if 'fg' in data.get('gt_fields', []):
            data['fg'] = functional.normalize(data['fg'], mean, std)
        if 'bg' in data.get('gt_fields', []):
            data['bg'] = functional.normalize(data['bg'], mean, std)

        return data


def reverse_transform(alpha: paddle.Tensor, trans_info: List[str]):
    """recover pred to origin shape"""
    for item in trans_info[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            alpha = F.interpolate(alpha, [h, w], mode='bilinear')
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            alpha = alpha[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return alpha

def save_alpha_pred(alpha: np.ndarray, trimap: np.ndarray = None):
    """
    The value of alpha is range [0, 1], shape should be [h,w]
    """
    if isinstance(trimap, str):
        trimap = cv2.imread(trimap, 0)
    alpha[trimap == 0] = 0
    alpha[trimap == 255] = 255
    alpha = (alpha).astype('uint8')
    return alpha


def cv2_to_base64(image: np.ndarray):
    """
    Convert data from BGR to base64 format.
    """
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str: str):
    """
    Convert data from base64 to BGR format.
    """
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data