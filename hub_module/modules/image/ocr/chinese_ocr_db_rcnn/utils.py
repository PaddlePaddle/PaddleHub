# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def draw_ocr(image,
             boxes,
             txts,
             scores,
             font_file,
             draw_txt=True,
             drop_score=0.5):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        draw.line([(box[0][0], box[0][1]), (box[1][0], box[1][1])], fill='red')
        draw.line([(box[1][0], box[1][1]), (box[2][0], box[2][1])], fill='red')
        draw.line([(box[2][0], box[2][1]), (box[3][0], box[3][1])], fill='red')
        draw.line([(box[3][0], box[3][1]), (box[0][0], box[0][1])], fill='red')
        draw.line([(box[0][0] - 1, box[0][1] + 1),
                   (box[1][0] - 1, box[1][1] + 1)],
                  fill='red')
        draw.line([(box[1][0] - 1, box[1][1] + 1),
                   (box[2][0] - 1, box[2][1] + 1)],
                  fill='red')
        draw.line([(box[2][0] - 1, box[2][1] + 1),
                   (box[3][0] - 1, box[3][1] + 1)],
                  fill='red')
        draw.line([(box[3][0] - 1, box[3][1] + 1),
                   (box[0][0] - 1, box[0][1] + 1)],
                  fill='red')

    if draw_txt:
        txt_color = (0, 0, 0)
        img = np.array(resize_img(img))
        _h = img.shape[0]
        blank_img = np.ones(shape=[_h, 600], dtype=np.int8) * 255
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)

        font_size = 20
        gap = 20
        title = "index           text           score"
        font = ImageFont.truetype(font_file, font_size, encoding="utf-8")

        draw_txt.text((20, 0), title, txt_color, font=font)
        count = 0
        for idx, txt in enumerate(txts):
            if scores[idx] < drop_score:
                continue
            font = ImageFont.truetype(font_file, font_size, encoding="utf-8")
            new_txt = str(count) + ':  ' + txt + '    ' + str(scores[count])
            draw_txt.text((20, gap * (count + 1)),
                          new_txt,
                          txt_color,
                          font=font)
            count += 1
        img = np.concatenate([np.array(img), np.array(blank_img)], axis=1)
    return img


def resize_img(img, input_size=600):
    img = np.array(img)
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return im


def get_image_ext(image):
    if image.shape[2] == 4:
        return ".png"
    return ".jpg"


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: x[0][1])
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
            (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes
