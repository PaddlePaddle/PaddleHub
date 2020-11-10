# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import List, Union

import cv2
import PIL
import numpy as np


def normalize(im: np.ndarray, mean: float, std: float) -> np.ndarray:
    '''
    Normalize the input image.

    Args:
        im(np.ndarray): Input image.
        mean(float): The mean value of normalization.
        std(float): The standard deviation value of normalization.
    '''
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def permute(im: np.ndarray) -> np.ndarray:
    '''
    Repermute the input image from [H, W, C] to [C, H, W].

    Args:
        im(np.ndarray): Input image.
    '''
    im = np.transpose(im, (2, 0, 1))
    return im


def resize(im: np.ndarray, target_size: Union[List[int], int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    '''
    Resize the input image.

    Args:
        im(np.ndarray): Input image.
        target_size(int|list[int]): The target size, if the input type is int, the target width and height will be set
                                    to this value, if the input type is list, the first element in the list represents
                                    the target width, and the second value represents the target height.
        interpolation(int): Interpolation method. Default to cv2.INTER_LINEAR.
    '''
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interpolation)
    return im


def resize_long(im: np.ndarray, long_size: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    '''
    Resize the long side of the input image to the target size.

    Args:
        im(np.ndarray): Input image.
        long_size(int|list[int]): The target size of long side.
        interpolation(int): Interpolation method. Default to cv2.INTER_LINEAR.
    '''
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(im, (resized_width, resized_height), interpolation=interpolation)
    return im


def horizontal_flip(im: np.ndarray) -> np.ndarray:
    '''
    Flip the picture horizontally.

    Args:
        im(np.ndarray): Input image.
    '''
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def vertical_flip(im: np.ndarray) -> np.ndarray:
    '''
    Flip the picture vertically.

    Args:
        im(np.ndarray): Input image.
    '''
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def brightness(im: np.ndarray, brightness_lower: float, brightness_upper: float) -> np.ndarray:
    '''
    Randomly disturb the brightness of the picture, user can use np.random.seed to fix the random behavior.

    Args:
        im(np.ndarray): Input image.
        brightness_lower(float): Lower bound of brightness.
        brightness_upper(float): Upper bound of brightness.
    '''
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = PIL.ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im: np.ndarray, contrast_lower: float, contrast_upper: float) -> np.ndarray:
    '''
    Randomly disturb the contrast of the picture, user can use np.random.seed to fix the random behavior.

    Args:
        im(np.ndarray): Input image.
        contrast_lower(float): Lower bound of contrast.
        contrast_upper(float): Upper bound of contrast.
    '''
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = PIL.ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im: np.ndarray, saturation_lower: float, saturation_upper: float) -> np.ndarray:
    '''
    Randomly disturb the saturation of the picture, user can use np.random.seed to fix the random behavior.

    Args:
        im(np.ndarray): Input image.
        saturation_lower(float): Lower bound of saturation.
        saturation_upper(float): Upper bound of saturation.
    '''
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = PIL.ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im: np.ndarray, hue_lower: float, hue_upper: float) -> np.ndarray:
    '''
    Randomly disturb the hue of the picture, user can use np.random.seed to fix the random behavior.

    Args:
        im(np.ndarray): Input image.
        hue_lower(float): Lower bound of hue.
        hue_upper(float): Upper bound of hue.
    '''
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = PIL.Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def rotate(im: np.ndarray, rotate_lower: float, rotate_upper: float) -> np.ndarray:
    '''
    Rotate the input image at random angle, user can use np.random.seed to fix the random behavior.

    Args:
        im(np.ndarray): Input image.
        rotate_lower(float): Lower bound of rotation angle.
        rotate_upper(float): Upper bound of rotation angle.
    '''
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im
