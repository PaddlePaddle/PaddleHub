# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import random
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
from paddle import fluid

__all__ = ['reader']


class DecodeImage(object):
    def __init__(self, to_rgb=True, with_mixup=False):
        """ Transform the image data to numpy format.

        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
        """
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup

    def __call__(self, im):
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im


class ResizeImage(object):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.

        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
        """
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.target_size = target_size

    def __call__(self, im):
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ValueError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)

        return im


class NormalizeImage(object):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, im):
        """Normalize the image.

        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im


class Permute(object):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.

        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, im):
        if self.channel_first:
            im = np.swapaxes(im, 1, 2)
            im = np.swapaxes(im, 1, 0)
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im


def reader(paths=[],
           images=None,
           decode_image=DecodeImage(to_rgb=True, with_mixup=False),
           resize_image=ResizeImage(
               target_size=512, interp=1, max_size=0, use_cv2=False),
           permute_image=Permute(to_bgr=False),
           normalize_image=NormalizeImage(
               mean=[104, 117, 123], std=[1, 1, 1], is_scale=False)):
    """
    data generator

    Args:
        paths (list[str]): paths to images.
        images (list(numpy.ndarray)): data of images, shape of each is [H, W, C]
        decode_image (class object): instance of <class 'DecodeImage' object>
        resize_image (class object): instance of <class 'ResizeImage' object>
        permute_image (class object): instance of <class 'Permute' object>
        normalize_image (class object): instance of <class 'NormalizeImage' object>
    """
    img_list = []
    if paths is not None:
        assert type(paths) is list, "type(paths) is not list."
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file path.".format(img_path)
            img = cv2.imread(img_path).astype('float32')
            img_list.append(img)
    if images is not None:
        for img in images:
            img_list.append(img)

    decode_image = DecodeImage(to_rgb=True, with_mixup=False)
    resize_image = ResizeImage(
        target_size=300, interp=1, max_size=0, use_cv2=False)
    permute_image = Permute()
    normalize_image = NormalizeImage(
        mean=[127.5, 127.5, 127.5],
        std=[127.502231, 127.502231, 127.502231],
        is_scale=False)

    for img in img_list:
        preprocessed_img = decode_image(img)
        preprocessed_img = resize_image(preprocessed_img)
        preprocessed_img = permute_image(preprocessed_img)
        preprocessed_img = normalize_image(preprocessed_img)
        yield [preprocessed_img]
