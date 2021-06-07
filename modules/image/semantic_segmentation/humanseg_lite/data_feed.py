# -*- coding:utf-8 -*-
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

__all__ = ['reader', 'preprocess_v']


def preprocess_v(img, w, h):
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR).astype(np.float32)
    img_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    img_std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    img = img.transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img


def reader(images=None, paths=None):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
        paths (list[str]): paths to images.

    Yield:
        each (collections.OrderedDict): info of original image, preprocessed image.
    """
    component = list()
    if paths:
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
            #print(im_path)
            im = cv2.imread(im_path).astype('float32')
            each['org_im'] = im
            each['org_im_path'] = im_path
            each['org_im_shape'] = im.shape
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}'.format(round(time.time(), 6) * 1e6)
            each['org_im_shape'] = im.shape
            component.append(each)

    for element in component:
        img = element['org_im'].copy()
        img = cv2.resize(img, (192, 192)).astype(np.float32)
        img_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
        img_std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
        img = img.transpose((2, 0, 1)) / 255
        img -= img_mean
        img /= img_std
        element['image'] = img
        yield element
