# coding=utf-8
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

__all__ = ['reader']


def preprocess(image):
    if image.mode == 'L':
        image = image.convert('RGB')
    shrink, max_shrink = get_shrink(image.size[1], image.size[0])
    image_shape = [3, image.size[1], image.size[0]]
    if shrink != 1:
        h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
        image = image.resize((w, h), Image.ANTIALIAS)
        image_shape = [3, h, w]

    img = np.array(image)
    img = to_chw_bgr(img)
    mean = [104., 117., 123.]
    scale = 0.007843
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img = img * scale
    img = np.array(img)
    return img


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.

    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


def get_shrink(height, width):
    """
    shrink the original image according to the org_im_height and org_im_width.
    calculate the value of shrink.

    Args:
        height (int): image height.
        width (int): image width.
    """
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width))**0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width))**0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5
    elif max_shrink <= 0.1:
        max_shrink = 0.1
    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


def reader(images, paths):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C], color space is BGR.
        paths (list[str]): paths to images.

    Yield:
        each (collections.OrderedDict): info of original image, preprocessed image.
    """
    component = list()
    if paths is not None:
        assert type(paths) is list, "paths should be a list."
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
            each['org_im'] = Image.open(im_path)
            each['org_im_width'], each['org_im_height'] = each['org_im'].size
            each['org_im_path'] = im_path
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            each['org_im_width'], each['org_im_height'] = each['org_im'].size
            each['org_im_path'] = 'ndarray_time={}'.format(round(time.time(), 6) * 1e6)
            component.append(each)

    for element in component:
        element['image'] = preprocess(element['org_im'])
        yield element
