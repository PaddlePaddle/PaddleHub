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

label_list = ['NO MASK', 'MASK']


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
    elif img.mode == 'L':  # black and white
        ext = '.jpg'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)

    return save_im_path


def draw_bounding_box_on_image(save_im_path, output_data):
    image = Image.open(save_im_path)
    draw = ImageDraw.Draw(image)
    for bbox in output_data:
        # draw bouding box
        if bbox['label'] == "MASK":
            draw.line([(bbox['left'], bbox['top']), (bbox['left'], bbox['bottom']), (bbox['right'], bbox['bottom']),
                       (bbox['right'], bbox['top']), (bbox['left'], bbox['top'])],
                      width=2,
                      fill='green')
        else:
            draw.line([(bbox['left'], bbox['top']), (bbox['left'], bbox['bottom']), (bbox['right'], bbox['bottom']),
                       (bbox['right'], bbox['top']), (bbox['left'], bbox['top'])],
                      width=2,
                      fill='red')
        # draw label
        text = bbox['label'] + ": %.2f%%" % (100 * bbox['confidence'])
        textsize_width, textsize_height = draw.textsize(text=text)
        if image.mode == 'RGB' or image.mode == 'RGBA':
            box_fill = (255, 255, 255)
            text_fill = (0, 0, 0)
        else:
            box_fill = (255)
            text_fill = (0)

        draw.rectangle(
            xy=(bbox['left'], bbox['top'] - (textsize_height + 5), bbox['left'] + textsize_width + 10, bbox['top'] - 3),
            fill=box_fill)
        draw.text(xy=(bbox['left'], bbox['top'] - 15), text=text, fill=text_fill)
    image.save(save_im_path)


def postprocess(confidence_out, org_im, org_im_path, detected_faces, output_dir, visualization):
    """
    Postprocess output of network. one element at a time.

    Args:
        confidence_out (numpy.ndarray): confidences of each label.
        org_im (numpy.ndarray): original image.
        org_im_path (list): path of original image.
        detected_faces (list): faces detected in a picture.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.

    Returns:
        output (dict): keys are 'data' and 'path', the correspoding values are:
            data (list[dict]): 6 keys, where
                'label' is `MASK` or `NO MASK`,
                'left', 'top', 'right', 'bottom' are the coordinate of detection bounding box,
                'confidence' is the confidence of mask detection.
            path (str): The path of original image.
    """
    output = dict()
    output['data'] = list()
    output['path'] = org_im_path

    for index, face in enumerate(detected_faces):
        label_idx = np.argmax(confidence_out[index])
        label_confidence = confidence_out[index][label_idx]
        bbox = dict()
        bbox['label'] = label_list[label_idx]
        bbox['confidence'] = label_confidence
        bbox['top'] = detected_faces[index]['top']
        bbox['bottom'] = detected_faces[index]['bottom']
        bbox['left'] = detected_faces[index]['left']
        bbox['right'] = detected_faces[index]['right']
        output['data'].append(bbox)

    if visualization:
        check_dir(output_dir)
        save_im_path = get_save_image_name(org_im, org_im_path, output_dir)
        cv2.imwrite(save_im_path, org_im)
        draw_bounding_box_on_image(save_im_path, output['data'])

    return output
