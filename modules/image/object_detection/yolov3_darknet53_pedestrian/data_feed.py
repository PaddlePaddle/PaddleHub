# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import cv2
import numpy as np

__all__ = ['reader']


def reader(paths=[], images=None):
    """
    data generator

    Args:
        paths (list[str]): paths to images.
        images (list(numpy.ndarray)): data of images, shape of each is [H, W, C]

    Yield:
        res (list): preprocessed image and the size of original image.
    """
    img_list = []
    if paths:
        assert type(paths) is list, "type(paths) is not list."
        for img_path in paths:
            assert os.path.isfile(img_path), "The {} isn't a valid file path.".format(img_path)
            img = cv2.imread(img_path).astype('float32')
            img_list.append(img)
    if images is not None:
        for img in images:
            img_list.append(img)

    for im in img_list:
        # im_size
        im_shape = im.shape
        im_size = np.array([im_shape[0], im_shape[1]], dtype=np.int32)

        # decode image
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # resize image
        target_size = 608
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('min size of image is 0')

        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)

        # normalize image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std

        # permute
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)

        yield [im, im_size]
