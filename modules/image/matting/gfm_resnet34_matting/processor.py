# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import base64

import cv2
import numpy as np
from paddleseg.transforms import functional


class ResizeByLong:
    """
    Resize the long side of an image to given size, and then scale the other side proportionally.

    Args:
        long_size (int): The target size of long side.
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, data):
        data = functional.resize_long(data, self.long_size)
        return data


class ResizeByShort:
    """
    Resize the short side of an image to given size, and then scale the other side proportionally.

    Args:
        short_size (int): The target size of short side.
    """

    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, data):
        
        data = functional.resize_short(data, self.short_size)
        
        return data

def gen_trimap_from_segmap_e2e(segmap):
	trimap = np.argmax(segmap, axis=1)[0]
	trimap = trimap.astype(np.int64)	
	trimap[trimap==1]=128
	trimap[trimap==2]=255
	return trimap.astype(np.uint8)

def get_masked_local_from_global_test(global_result, local_result):
	weighted_global = np.ones(global_result.shape)
	weighted_global[global_result==255] = 0
	weighted_global[global_result==0] = 0
	fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
	return fusion_result

def cv2_to_base64(image: np.ndarray):
    """
    Convert data from BGR to base64 format.
    """
    data = cv2.imencode('.png', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str: str):
    """
    Convert data from base64 to BGR format.
    """
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data