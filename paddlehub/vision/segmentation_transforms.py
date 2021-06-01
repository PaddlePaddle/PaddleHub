# coding: utf8
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

import os
import random
from typing import Callable, Union, List, Tuple

import cv2
import numpy as np
from PIL import Image
import paddlehub.vision.functional as F


class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms: Callable, to_rgb: bool = True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                             'must be equal or larger than 1!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im: Union[np.ndarray, str], label: Union[np.ndarray, str] = None) -> Tuple:
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = np.asarray(Image.open(label))
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = np.transpose(im, (2, 0, 1))
        return (im, label)


class ColorMap:
    "Calculate color map for mapping segmentation result."

    def __init__(self, num_classes: int = 256):
        self.num_classes = num_classes + 1

    def __call__(self) -> np.ndarray:
        color_map = self.num_classes * [0, 0, 0]
        for i in range(0, self.num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        color_map = color_map[1:]
        return color_map


class SegmentVisual:
    """Visualization the segmentation result.
    Args:
        weight(float): weight of original image in combining image, default is 0.6.
    """

    def __init__(self, weight: float = 0.6):
        self.weight = weight
        self.get_color_map_list = ColorMap(256)

    def __call__(self, image: str, result: np.ndarray, save_dir: str) -> np.ndarray:
        color_map = self.get_color_map_list()
        color_map = np.array(color_map).astype("uint8")
        # Use OpenCV LUT for color mapping
        c1 = cv2.LUT(result, color_map[:, 0])
        c2 = cv2.LUT(result, color_map[:, 1])
        c3 = cv2.LUT(result, color_map[:, 2])
        pseudo_img = np.dstack((c1, c2, c3))
        im = cv2.imread(image)
        vis_result = cv2.addWeighted(im, self.weight, pseudo_img, 1 - self.weight, 0)

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image_name = os.path.split(image)[-1]
            out_path = os.path.join(save_dir, image_name)
            cv2.imwrite(out_path, vis_result)

        return vis_result


class Padding:
    """
    Add bottom-right padding to a raw image or annotation image.
    Args:
        target_size (list|tuple): The target size after padding.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    Raises:
        TypeError: When target_size is neither list nor tuple.
        ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size: Union[List[int], Tuple[int], int],
                 im_padding_value: Union[List[int], Tuple[int], int] = (128, 128, 128),
                 label_padding_value: int = 255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError('`target_size` should include 2 elements, but it is {}'.format(target_size))
        else:
            raise TypeError("Type of target_size is invalid. It should be list or tuple, now is {}".format(
                type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im: np.ndarray, label: np.ndarray = None) -> Tuple:
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im_height, im_width = im.shape[0], im.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'The size of image should be less than `target_size`, but the size of image ({}, {}) is larger than `target_size` ({}, {})'
                .format(im_width, im_height, target_width, target_height))
        else:
            im = cv2.copyMakeBorder(im, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=self.im_padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(
                    label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=self.label_padding_value)
        if label is None:
            return (im, )
        else:
            return (im, label)


class Normalize:
    """
    Normalize an image.
    Args:
        mean (list|tuple): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list|tuple): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].
    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self,
                 mean: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
                 std: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple)) and isinstance(self.std, (list, tuple))):
            raise ValueError("{}: input type is invalid. It should be list or tuple".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im: np.ndarray, label: np.ndarray = None) -> Tuple:
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = F.normalize(im, mean, std)

        if label is None:
            return (im, )
        else:
            return (im, label)


class Resize:
    """
    Resize an image.

    Args:
        target_size (list|tuple, optional): The target size of image. Default: (512, 512).
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".

    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size: Union[List[int], Tuple[int]] = (512, 512), interp: str = 'LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError('`target_size` should include 2 elements, but it is {}'.format(target_size))
        else:
            raise TypeError("Type of `target_size` is invalid. It should be list or tuple, but it is {}".format(
                type(target_size)))

        self.target_size = target_size

    def __call__(self, im: np.ndarray, label: np.ndarray = None) -> Tuple:
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label),

        Raises:
            TypeError: When the 'img' type is not numpy.
            ValueError: When the length of "im" shape is not 3.
        """

        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = F.resize(im, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = F.resize(label, self.target_size, cv2.INTER_NEAREST)

        if label is None:
            return (im, )
        else:
            return (im, label)
