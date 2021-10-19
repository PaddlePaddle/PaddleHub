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
    Preprocess to get image data.

    Args:
        images (list): list of dict objects, each dict contains key:
            content(str): value is a numpy.ndarry with shape [H, W, C], content data.
            styles(str): value is a list of numpy.ndarray with shape [H, W, C], styles data.
            weights(str, optional): value is the interpolation weights correspond to styles.
        paths (list): list of dict objects, each dict contains key:
            content(str): value is the path to content.
            styles(str): value is the paths to styles.
            weights(str, optional): value is the interpolation weights correspond to styles.
    Yield:
        im (numpy.ndarray): preprocessed data, with shape (1, 3, 512, 512).
    """
    pipeline_list = list()
    # images
    for key, data in [('im_arr', images), ('im_path', paths)]:
        if data is not None:
            for component in data:
                each_res = OrderedDict()
                # content_arr
                each_res['content_arr'], w, h = _handle_single(**{key: component['content']})
                # styles_arr_list
                styles_list = component['styles']
                styles_num = len(styles_list)
                each_res['styles_arr_list'] = []
                for i, style_arr in enumerate(styles_list):
                    each_res['styles_arr_list'].append(_handle_single(**{key: style_arr})[0])
                # style_interpolation_weights
                if 'weights' in component:
                    assert len(component['weights']
                               ) == styles_num, "The number of weights must be equal to the number of styles."
                    each_res['style_interpolation_weights'] = component['weights']
                else:
                    each_res['style_interpolation_weights'] = np.ones(styles_num)
                each_res['style_interpolation_weights'] = [
                    each_res['style_interpolation_weights'][j] / sum(each_res['style_interpolation_weights'])
                    for j in range(styles_num)
                ]
                pipeline_list.append([each_res, w, h])

    # yield
    for element in pipeline_list:
        yield element


def _handle_single(im_path=None, im_arr=None):
    """
    Preprocess to get image data.
    Args:
        im_path (str): path to image.
        im_arr (numpy.ndarray): image data, with shape (H, W, 3).
    Returns:
        im (numpy.ndarray): preprocessed data, with shape (1, 3, 512, 512).
    """
    im = None
    if im_path is not None:
        im = cv2.imread(im_path)
        if im is None:
            raise FileNotFoundError('Error: The file path "{}"  may not exist or is not a valid image file, please provide a valid path.'.format(im_path))
        else:
            assert(len(im.shape) == 3, 'The input image shape should be [H, W, 3], but got {}'.format(im.shape))
            assert(im.shape[2] == 3,  'The input image should have 3 channels, but got {}'.format(im.shape[2]))
            im = im[:, :, ::-1].astype(np.float32)    ### Image should have 3-channels, and BGR format is arranged by cv2, we should change it to RGB.
    if im_arr is not None:
        im = im_arr[:, :, ::-1].astype(np.float32)
    if im is None:
        raise ValueError('No image data is provided. Please check the input "images" and "paths".')
    w, h = im.shape[1], im.shape[0]
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR)
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im /= 255.0
    return im, w, h
