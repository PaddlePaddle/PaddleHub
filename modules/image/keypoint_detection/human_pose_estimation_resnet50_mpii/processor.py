# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os
import time
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['base64_to_cv2', 'postprocess']


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def get_max_preds(batch_heatmaps):
    """
    Get predictions from score maps.

    Args:
        batch_heatmaps (numpy.ndarray): output of the network, with shape [N, C, H, W]
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return preds, maxvals


def predict_results(batch_heatmaps):
    batch_size, num_joints, heatmap_height, heatmap_width = batch_heatmaps.shape
    preds, maxvals = get_max_preds(batch_heatmaps)
    return preds[0] * 4, num_joints


def postprocess(out_heatmaps, org_im, org_im_shape, org_im_path, output_dir, visualization):
    """
    Postprocess output of network. one image at a time.

    Args:
        out_heatmaps (numpy.ndarray): output of network.
        org_im (numpy.ndarray): original image.
        org_im_shape (list): shape pf original image.
        org_im_path (list): path of riginal image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.

    Returns:
        res (dict): Output of postprocess. keys contains 'path', 'data', the corresponding valus are:
            path (str): the path of original image.
            data (OrderedDict): The key points of human pose.
    """
    res = dict()
    res['path'] = org_im_path
    res['data'] = OrderedDict()
    preds, num_joints = predict_results(out_heatmaps)
    scale_horizon = org_im_shape[1] * 1.0 / 384
    scale_vertical = org_im_shape[0] * 1.0 / 384
    preds = np.multiply(preds, (scale_horizon, scale_vertical)).astype(int)
    if visualization:
        icolor = (255, 137, 0)
        ocolor = (138, 255, 0)
        rendered_im = org_im.copy()
        for j in range(num_joints):
            x, y = preds[j]
            cv2.circle(rendered_im, (x, y), 3, icolor, -1, 16)
            cv2.circle(rendered_im, (x, y), 6, ocolor, 1, 16)
        check_dir(output_dir)
        save_im_name = get_save_image_name(org_im, org_im_path, output_dir)
        cv2.imwrite(save_im_name, rendered_im)
        print('image saved in {}'.format(save_im_name))

    # articulation
    preds = list(map(lambda pred: [int(_) for _ in pred], preds))
    res['data']['left_ankle'] = list(preds[0])
    res['data']['left_knee'] = list(preds[1])
    res['data']['left_hip'] = list(preds[2])
    res['data']['right_hip'] = list(preds[3])
    res['data']['right_knee'] = list(preds[4])
    res['data']['right_ankle'] = list(preds[5])
    res['data']['pelvis'] = list(preds[6])
    res['data']['thorax'] = list(preds[7])
    res['data']['upper_neck'] = list(preds[8])
    res['data']['head_top'] = list(preds[9])
    res['data']['right_wrist'] = list(preds[10])
    res['data']['right_elbow'] = list(preds[11])
    res['data']['right_shoulder'] = list(preds[12])
    res['data']['left_shoulder'] = list(preds[13])
    res['data']['left_elbow'] = list(preds[14])
    res['data']['left_wrist'] = list(preds[15])

    return res


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
    ext = '.jpg'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)

    return save_im_path
