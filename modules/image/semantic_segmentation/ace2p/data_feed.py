# coding=utf-8
import os
import time
from collections import OrderedDict

import cv2
import numpy as np

from ace2p.processor import get_direction, get_3rd_point, get_affine_transform

__all__ = ['reader']


def _box2cs(box, aspect_ratio):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio)


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std=200):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    return center, scale


def preprocess(org_im, scale, rotation):
    image = org_im.copy()
    image_height, image_width, _ = image.shape

    aspect_ratio = scale[1] * 1.0 / scale[0]
    image_center, image_scale = _box2cs([0, 0, image_width - 1, image_height - 1], aspect_ratio)

    trans = get_affine_transform(image_center, image_scale, rotation, scale)
    image = cv2.warpAffine(
        image,
        trans, (int(scale[1]), int(scale[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    img_mean = np.array([0.406, 0.456, 0.485]).reshape((1, 1, 3))
    img_std = np.array([0.225, 0.224, 0.229]).reshape((1, 1, 3))
    image = image.astype(np.float)
    image = (image / 255.0 - img_mean) / img_std
    image = image.transpose(2, 0, 1).astype(np.float32)

    image_info = {
        'image_center': image_center,
        'image_height': image_height,
        'image_width': image_width,
        'image_scale': image_scale,
        'rotation': rotation,
        'scale': scale
    }

    return image, image_info


def reader(images, paths, scale, rotation):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
        paths (list[str]): paths to images.
        scale (tuple): size of preprocessed image.
        rotation (int): rotation angle, used for obtaining affine matrix in preprocess.

    Yield:
        element (collections.OrderedDict): info of original image and preprocessed image.
    """
    component = list()
    if paths:
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
            each['org_im_path'] = 'ndarray_time={}.jpg'.format(round(time.time(), 6) * 1e6)
            component.append(each)

    for element in component:
        element['image'], element['image_info'] = preprocess(element['org_im'], scale, rotation)
        yield element
