# coding=utf-8
import os
import time
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['reader']


def preprocess(org_im, shrink):
    image = org_im.copy()
    image_height, image_width, image_channel = image.shape
    if shrink != 1:
        image_height, image_width = int(image_height * shrink), int(image_width * shrink)
        image = cv2.resize(image, (image_width, image_height), cv2.INTER_NEAREST)
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # mean, std
    mean = [104., 117., 123.]
    scale = 0.007843
    image = image.astype('float32')
    image -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    image = image * scale
    return image, image_height, image_width


def reader(images, paths, shrink):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C], color space is BGR.
        paths (list[str]): paths to images.
        shrink (float): parameter to control the resize scale in preprocess.

    Yield:
        each (collections.OrderedDict): info of original image, preprocessed image.
    """
    component = list()
    if paths is not None:
        assert type(paths) is list, "paths should be a list."
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
            im = cv2.imread(im_path)
            each['org_im'] = im
            each['org_im_path'] = im_path
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}'.format(round(time.time(), 6) * 1e6)
            component.append(each)

    for element in component:
        element['image'], element['image_height'], element['image_width'] = preprocess(element['org_im'], shrink)
        yield element
