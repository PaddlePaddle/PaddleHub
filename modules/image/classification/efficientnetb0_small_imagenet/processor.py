# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64

import cv2
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
