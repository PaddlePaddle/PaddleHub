# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
import inspect
import math
import os
import random
import sys
from functools import partial

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
import six
from paddle.vision.transforms import ColorJitter as RawColorJitter
from PIL import Image


def create_operators(params, class_num=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    current_module = sys.modules[__name__]
    for operator in params:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op_func = getattr(current_module, op_name)
        if "class_num" in inspect.getfullargspec(op_func).args:
            param.update({"class_num": class_num})
        op = op_func(**param)
        ops.append(op)

    return ops


class UnifiedResize(object):

    def __init__(self, interpolation=None, backend="cv2"):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }

        def _pil_resize(src, size, resample):
            pil_img = Image.fromarray(src)
            pil_img = pil_img.resize(size, resample)
            return np.asarray(pil_img)

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2.resize, interpolation=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation)
        else:
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        return self.resize_func(src, size)


class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass


class DecodeImage(object):
    """ decode image """

    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.to_rgb = to_rgb
        self.to_np = to_np  # to numpy
        self.channel_first = channel_first  # only enabled when to_np is True

    def __call__(self, img):
        if six.PY2:
            assert type(img) is str and len(img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


class ResizeImage(object):
    """ resize image """

    def __init__(self, size=None, resize_short=None, interpolation=None, backend="cv2"):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(interpolation=interpolation, backend=backend)

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class RandCropImage(object):
    """ random crop image """

    def __init__(self, size, scale=None, ratio=None, interpolation=None, backend="cv2"):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

        self._resize_func = UnifiedResize(interpolation=interpolation, backend=backend)

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2), (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]

        return self._resize_func(img, size)


class RandFlipImage(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            return cv2.flip(img, self.flip_code)
        else:
            return img


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', output_fp16=False, channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [3, 4], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros((1, img_h, img_w)) if self.order == 'chw' else np.zeros((img_h, img_w, 1))
            img = (np.concatenate((img, pad_zeros), axis=0) if self.order == 'chw' else np.concatenate(
                (img, pad_zeros), axis=2))
        return img.astype(self.output_dtype)


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))


class ColorJitter(RawColorJitter):
    """ColorJitter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)
        img = super()._apply_image(img)
        if isinstance(img, Image.Image):
            img = np.asarray(img)
        return img


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


class Topk(object):

    def __init__(self, topk=1, class_id_map_file=None):
        assert isinstance(topk, (int, ))
        self.class_id_map = self.parse_class_id_map(class_id_map_file)
        self.topk = topk

    def parse_class_id_map(self, class_id_map_file):
        if class_id_map_file is None:
            return None
        if not os.path.exists(class_id_map_file):
            print(
                "Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
            )
            return None

        try:
            class_id_map = {}
            with open(class_id_map_file, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    partition = line.split("\n")[0].partition(" ")
                    class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            print(ex)
            class_id_map = None
        return class_id_map

    def __call__(self, x, file_names=None, multilabel=False):
        assert isinstance(x, paddle.Tensor)
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        x = F.softmax(x, axis=-1) if not multilabel else F.sigmoid(x)
        x = x.numpy()
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32") if not multilabel else np.where(
                probs >= 0.5)[0].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                "class_ids": clas_id_list,
                "scores": np.around(score_list, decimals=5).tolist(),
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            if label_name_list is not None:
                result["label_names"] = label_name_list
            y.append(result)
        return y
