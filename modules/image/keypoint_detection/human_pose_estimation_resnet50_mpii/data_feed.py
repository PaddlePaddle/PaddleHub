# coding=utf-8
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

__all__ = ['reader']


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
        im = element['org_im'].copy()
        im = cv2.resize(im, (384, 384))
        im = im.astype('float32')
        im = im.transpose((2, 0, 1)) / 255
        im -= np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        im /= np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        element['image'] = im
        yield element
