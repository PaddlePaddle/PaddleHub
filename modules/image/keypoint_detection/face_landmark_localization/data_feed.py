# coding=utf-8
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

__all__ = ['reader']


def reader(face_detector, images=None, paths=None, use_gpu=False):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C].
        paths (list[str]): paths to images.

    Yield:
        each (collections.OrderedDict): info of original image, preprocessed image.
    """
    components = []
    if paths:
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
            im = cv2.imread(im_path)
            each['orig_im'] = im
            each['orig_im_shape'] = im.shape
            each['orig_im_path'] = im_path
            components.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['orig_im'] = im
            each['orig_im_path'] = None
            each['orig_im_shape'] = im.shape
            components.append(each)

    for idx, item in enumerate(
            face_detector.face_detection(
                images=[component['orig_im'] for component in components], use_gpu=use_gpu, visualization=False)):
        for face in item['data']:
            width = int(components[idx]['orig_im_shape'][1])
            height = int(components[idx]['orig_im_shape'][0])
            x1 = 0 if int(face['left']) < 0 else int(face['left'])
            x2 = width if int(face['right']) > width else int(face['right'])
            y1 = 0 if int(face['top']) < 0 else int(face['top'])
            y2 = height if int(face['bottom']) > height else int(face['bottom'])
            roi = components[idx]['orig_im'][y1:y2 + 1, x1:x2 + 1, :]
            gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray_img = cv2.resize(gray_img, (60, 60), interpolation=cv2.INTER_CUBIC)
            mean, std_dev = cv2.meanStdDev(gray_img)
            gray_img = (gray_img - mean[0][0]) / (0.000001 + std_dev[0][0])
            gray_img = np.expand_dims(gray_img, axis=0)
            yield {
                'face': gray_img,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'orig_im': components[idx]['orig_im'],
                'orig_im_path': components[idx]['orig_im_path'],
                'orig_im_shape': components[idx]['orig_im_shape'],
                'id': idx
            }
