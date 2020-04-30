# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import cv2
import os

import numpy as np


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def postprocess(data_out, label_list, top_k):
    """
    Postprocess output of network, one image at a time.

    Args:
        data_out (numpy.ndarray): output data of network.
        label_list (list): list of label.
        top_k (int): Return top k results.
    """
    output = []
    for result in data_out:
        result_i = softmax(result)
        output_i = {}
        indexs = np.argsort(result_i)[::-1][0:top_k]
        for index in indexs:
            label = label_list[index].split(',')[0]
            output_i[label] = float(result_i[index])
        output.append(output_i)
    return output
