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
from PIL import Image

__all__ = ['base64_to_cv2', 'postprocess']


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def check_dir(dir_path):
    """
    Create directory to save processed image.

    Args:
        dir_path (str): directory path to save images.

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def get_save_image_name(org_im, org_im_path, output_dir):
    """
    Get save image name from source image path.
    """
    # name prefix of original image
    org_im_name = os.path.split(org_im_path)[-1]
    im_prefix = os.path.splitext(org_im_name)[0]
    # extension
    img = Image.fromarray(org_im[:, :, ::-1])
    if img.mode == 'RGBA':
        ext = '.png'
    elif img.mode == 'RGB':
        ext = '.jpg'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)

    return save_im_path


def clip_bbox(bbox, img_height, img_width):
    bbox['left'] = int(max(min(bbox['left'], img_width), 0.))
    bbox['top'] = int(max(min(bbox['top'], img_height), 0.))
    bbox['right'] = int(max(min(bbox['right'], img_width), 0.))
    bbox['bottom'] = int(max(min(bbox['bottom'], img_height), 0.))
    return bbox


def postprocess(data_out, org_im, org_im_path, image_height, image_width, output_dir, visualization, shrink,
                confs_threshold):
    """
    Postprocess output of network. one image at a time.

    Args:
        data_out (numpy.ndarray): output of network.
        org_im (numpy.ndarray): original image.
        org_im_path (list): path of riginal image.
        image_height (int): height of preprocessed image.
        image_width (int): width of preprocessed image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
        shrink (float): parameter to control the resize scale in preprocess.
        confs_threshold (float): confidence threshold.

    Returns:
        output (dict): keys are 'data' and 'path', the correspoding values are:
            data (list[dict]): 5 keys, where
                'left', 'top', 'right', 'bottom' are the coordinate of detection bounding box,
                'confidence' is the confidence this bbox.
            path (str): The path of original image.
    """
    output = dict()
    output['data'] = list()
    output['path'] = org_im_path

    for each_data in data_out:
        # each_data is a list: [label, confidence, left, top, right, bottom]
        if each_data[1] > confs_threshold:
            dt_bbox = dict()
            dt_bbox['confidence'] = float(each_data[1])
            dt_bbox['left'] = image_width * each_data[2] / shrink
            dt_bbox['top'] = image_height * each_data[3] / shrink
            dt_bbox['right'] = image_width * each_data[4] / shrink
            dt_bbox['bottom'] = image_height * each_data[5] / shrink
            dt_bbox = clip_bbox(dt_bbox, org_im.shape[0], org_im.shape[1])
            output['data'].append(dt_bbox)

    if visualization:
        check_dir(output_dir)
        save_im_path = get_save_image_name(org_im, org_im_path, output_dir)
        im_out = org_im.copy()
        if len(output['data']) > 0:
            for bbox in output['data']:
                cv2.rectangle(im_out, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), (255, 255, 0), 2)
        cv2.imwrite(save_im_path, im_out)

    return output
