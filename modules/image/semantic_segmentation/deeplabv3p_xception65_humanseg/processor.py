# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import OrderedDict

import base64
import cv2
import numpy as np

__all__ = ['cv2_to_base64', 'base64_to_cv2', 'postprocess']


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def postprocess(data_out, org_im, org_im_shape, org_im_path, output_dir, visualization, thresh=120):
    """
    Postprocess output of network. one image at a time.

    Args:
        data_out (numpy.ndarray): output of network.
        org_im (numpy.ndarray): original image.
        org_im_shape (list): shape pf original image.
        org_im_path (list): path of riginal image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
        thresh (float): threshold.

    Returns:
        result (dict): The data of processed image.
    """
    result = dict()
    for logit in data_out:
        logit = logit[1] * 255
        logit = cv2.resize(logit, (org_im_shape[1], org_im_shape[0]))
        logit -= thresh
        logit[logit < 0] = 0
        logit = 255 * logit / (255 - thresh)
        rgba = np.concatenate((org_im, np.expand_dims(logit, axis=2)), axis=2)

        if visualization:
            check_dir(output_dir)
            save_im_path = get_save_image_name(org_im, org_im_path, output_dir)
            cv2.imwrite(save_im_path, rgba)
            result['save_path'] = save_im_path
            result['data'] = rgba[:, :, 3]
        else:
            result['data'] = rgba[:, :, 3]
    return result


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def get_save_image_name(org_im, org_im_path, output_dir):
    """
    Get save image name from source image path.
    """
    # name prefix of orginal image
    org_im_name = os.path.split(org_im_path)[-1]
    im_prefix = os.path.splitext(org_im_name)[0]
    ext = '.png'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)

    return save_im_path
