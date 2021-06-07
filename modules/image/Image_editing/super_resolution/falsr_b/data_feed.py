# -*- coding:utf-8 -*-
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
            im = im.astype(np.float32)
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}'.format(round(time.time(), 6) * 1e6)
            each['org_im_shape'] = im.shape
            component.append(each)

    for element in component:
        img = element['org_im'].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        shape = img.shape
        img_scale = cv2.resize(img, (shape[1] * 2, shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        img_y = np.expand_dims(img[:, :, 0], axis=2)
        img_scale_pbpr = img_scale[..., 1:]
        img_y = img_y.transpose((2, 0, 1)) / 255
        img_scale_pbpr = img_scale_pbpr.transpose(2, 0, 1) / 255
        element['img_y'] = img_y
        element['img_scale_pbpr'] = img_scale_pbpr
        yield element


if __name__ == "__main__":
    path = ['BSD100_001.png']
    reader(paths=path)
