# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from paddle import fluid

__all__ = ['test_reader']


def test_reader(paths=None, images=None):
    """
    data generator

    Args:
        paths (list[str]): paths to images.
        images (list(numpy.ndarray)): data of images, shape of each is [H, W, C]

    Yield:
        res (dict): key contains 'image', 'im_info', 'im_shape', the corresponding values is:
            image (numpy.ndarray): the image to be fed into network
            im_info (numpy.ndarray): the info about the preprocessed.
            im_shape (numpy.ndarray): the shape of image.
    """
    img_list = list()
    if paths:
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file path.".format(img_path)
            img = cv2.imread(img_path).astype('float32')
            img_list.append(img)
    if images is not None:
        for img in images:
            img_list.append(img)

    for im in img_list:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32, copy=False)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std

        target_size = 800
        max_size = 1333

        shape = im.shape
        # im_shape holds the original shape of image.
        im_shape = np.array([shape[0], shape[1], 1.0]).astype('float32')
        im_size_min = np.min(shape[0:2])
        im_size_max = np.max(shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        resize_w = np.round(im_scale * float(shape[1]))
        resize_h = np.round(im_scale * float(shape[0]))
        # im_info holds the resize info of image.
        im_info = np.array([resize_h, resize_w, im_scale]).astype('float32')

        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR)

        # HWC --> CHW
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)
        yield {'image': im, 'im_info': im_info, 'im_shape': im_shape}


def padding_minibatch(batch_data, coarsest_stride=0, use_padded_im_info=True):
    max_shape_org = np.array(
        [data['image'].shape for data in batch_data]).max(axis=0)
    if coarsest_stride > 0:
        max_shape = np.zeros((3)).astype('int32')
        max_shape[1] = int(
            np.ceil(max_shape_org[1] / coarsest_stride) * coarsest_stride)
        max_shape[2] = int(
            np.ceil(max_shape_org[2] / coarsest_stride) * coarsest_stride)
    else:
        max_shape = max_shape_org.astype('int32')

    padding_image = list()
    padding_info = list()
    padding_shape = list()

    for data in batch_data:
        im_c, im_h, im_w = data['image'].shape
        # image
        padding_im = np.zeros((im_c, max_shape[1], max_shape[2]),
                              dtype=np.float32)
        padding_im[:, 0:im_h, 0:im_w] = data['image']
        padding_image.append(padding_im)
        # im_info
        data['im_info'][
            0] = max_shape[1] if use_padded_im_info else max_shape_org[1]
        data['im_info'][
            1] = max_shape[2] if use_padded_im_info else max_shape_org[2]
        padding_info.append(data['im_info'])
        padding_shape.append(data['im_shape'])

    padding_image = np.array(padding_image).astype('float32')
    padding_info = np.array(padding_info).astype('float32')
    padding_shape = np.array(padding_shape).astype('float32')
    return padding_image, padding_info, padding_shape
