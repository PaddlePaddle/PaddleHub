# -*- coding:utf-8 -*-
import os
import time
import base64

import cv2
from PIL import Image
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


def postprocess(data_out, org_im, org_im_path, output_dir, visualization, thresh=120):
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
    for i, img in enumerate(data_out):

        img = np.squeeze(img[0].as_ndarray(), 0).transpose((1, 2, 0))
        img = ((img + 1) * 127.5).astype(np.uint8)
        img = cv2.resize(img, (256, 341), cv2.INTER_CUBIC)
        fake_image = Image.fromarray(img)

        if visualization:
            check_dir(output_dir)
            save_im_path = get_save_image_name(org_im_path, output_dir, i)
            img_name = '{}.png'.format(i)
            fake_image.save(os.path.join(output_dir, img_name))

        result['data_{}'.format(i)] = img

    return result


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def get_save_image_name(org_im_path, output_dir, num):
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
        save_im_path = os.path.join(output_dir, im_prefix + str(num) + ext)

    return save_im_path
