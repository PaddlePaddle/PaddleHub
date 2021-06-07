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
from PIL import Image, ImageDraw

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


def get_save_image_name(img, org_im_path, output_dir):
    """
    Get save image name.
    """
    # name prefix of original image
    org_im_name = os.path.split(org_im_path)[-1]
    im_prefix = os.path.splitext(org_im_name)[0]
    # extension
    if img.mode == 'RGBA':
        ext = '.png'
    else:
        ext = '.jpg'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)
    return save_im_path


def draw_bboxes(image, bboxes, org_im_path, output_dir):
    """
    Draw bounding boxes on image.

    Args:
        bboxes (np.array): bounding boxes.
    """
    draw = ImageDraw.Draw(image)
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=4, fill='red')
    save_name = get_save_image_name(image, org_im_path, output_dir)
    image.save(save_name)


def postprocess(data_out, org_im, org_im_path, org_im_width, org_im_height, output_dir, visualization, score_thresh):
    """
    Postprocess output of network. one image at a time.

    Args:
        data_out (numpy.ndarray): output of network.
        org_im: (PIL.Image object): original image.
        org_im_path (str): path of original image.
        org_im_width (int): width of original image.
        org_im_height (int): height of original image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.

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

    if data_out.shape[0] == 0:
        print("No face detected in {}".format(org_im_path))
    else:
        det_conf = data_out[:, 1]
        det_xmin = org_im_width * data_out[:, 2]
        det_ymin = org_im_height * data_out[:, 3]
        det_xmax = org_im_width * data_out[:, 4]
        det_ymax = org_im_height * data_out[:, 5]
        dets = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
        keep_index = np.where(dets[:, 4] >= score_thresh)[0]
        dets = dets[keep_index, :]

        if dets.shape[0] == 0:
            print("No face detected in {}".format(org_im_path))
        else:
            for detect_face in dets:
                dt_i = dict()
                dt_i['left'] = float(detect_face[0])
                dt_i['top'] = float(detect_face[1])
                dt_i['right'] = float(detect_face[2])
                dt_i['bottom'] = float(detect_face[3])
                dt_i['confidence'] = float(detect_face[4])
                output['data'].append(dt_i)

        if visualization:
            check_dir(output_dir)
            draw_bboxes(org_im, dets[:, 0:4], org_im_path, output_dir)
    return output
