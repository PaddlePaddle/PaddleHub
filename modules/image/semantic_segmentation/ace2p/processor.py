# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import base64
import cv2
import numpy as np
from PIL import Image

__all__ = ['cv2_to_base64', 'base64_to_cv2', 'get_palette', 'postprocess']


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


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
    # name prefix of orginal image
    org_im_name = os.path.split(org_im_path)[-1]
    im_prefix = os.path.splitext(org_im_name)[0]
    ext = '.png'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)

    return save_im_path


def get_direction(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list) and not isinstance(scale, tuple):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]
    rot_rad = np.pi * rot / 180
    src_direction = get_direction([0, src_w * -0.5], rot_rad)
    dst_direction = np.array([0, (dst_w - 1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_direction + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_direction
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans, (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)
    return target_logits


def get_palette(num_cls):
    """
    Returns the color map for visualizing the segmentation mask.

    Args:
        num_cls: Number of classes

    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def postprocess(data_out, org_im, org_im_path, image_info, output_dir, visualization, palette):
    """
    Postprocess output of network. one image at a time.

    Args:
        data_out (numpy.ndarray): output of neural network.
        org_im (numpy.ndarray): orginal image.
        org_im_path (str): path of original image.
        image_info (dict): info about the preprocessed image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
        palette (list): The palette to draw.

    Returns:
        res (list[dict]): keys contain 'path', 'data', the corresponding value is:
            path (str): The path of original image.
            data (numpy.ndarray): The postprocessed image data, only the alpha channel.
    """
    result = dict()
    result['path'] = org_im_path

    image_center = image_info['image_center']
    image_scale = image_info['image_scale']
    image_width = image_info['image_width']
    image_height = image_info['image_height']
    scale = image_info['scale']

    data_out = np.squeeze(data_out)
    data_out = np.transpose(data_out, [1, 2, 0])
    logits_result = transform_logits(data_out, image_center, image_scale, image_width, image_height, scale)
    parsing = np.argmax(logits_result, axis=2)
    parsing_im = np.asarray(parsing, dtype=np.uint8)
    result['data'] = parsing_im

    if visualization:
        check_dir(output_dir)
        save_im_path = get_save_image_name(org_im, org_im_path, output_dir)
        parsing_im = Image.fromarray(parsing_im)
        parsing_im.putpalette(palette)
        parsing_im.save(save_im_path)

    return result
