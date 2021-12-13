# -*- coding:utf-8 -*-
import os
import time
from collections import OrderedDict

from PIL import Image, ImageOps
import numpy as np
from PIL import Image
import cv2

__all__ = ['reader']


def reader(images=None, paths=None, org_labels=None, target_labels=None):
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
        for i, im_path in enumerate(paths):
            each = OrderedDict()
            assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
            im = cv2.imread(im_path)
            each['org_im'] = im
            each['org_im_path'] = im_path
            each['org_label'] = np.array(org_labels[i]).astype('float32')
            if not target_labels:
                each['target_label'] = np.array(org_labels[i]).astype('float32')
            else:
                each['target_label'] = np.array(target_labels[i]).astype('float32')
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for i, im in enumerate(images):
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}'.format(round(time.time(), 6) * 1e6)
            each['org_label'] = np.array(org_labels[i]).astype('float32')
            if not target_labels:
                each['target_label'] = np.array(org_labels[i]).astype('float32')
            else:
                each['target_label'] = np.array(target_labels[i]).astype('float32')
            component.append(each)

    for element in component:
        img = cv2.cvtColor(element['org_im'], cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = (img.astype('float32') / 255.0 - 0.5) / 0.5
        img = img.transpose([2, 0, 1])
        element['img'] = img[np.newaxis, :, :, :]

        yield element
