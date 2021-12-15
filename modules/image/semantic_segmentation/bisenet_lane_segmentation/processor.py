import base64
import collections.abc
from itertools import combinations
from typing import Union, List, Tuple, Callable

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F


def get_reverse_list(ori_shape: list, transforms: Callable) -> list:
    """
    get reverse list of transform.

    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.

    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    h, w = ori_shape[0], ori_shape[1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['Crop']:
            reverse_list.append(('crop', (op.up_h_off, op.down_h_off),
                                 (op.left_w_off, op.right_w_off)))
            h = h - op.up_h_off
            h = h - op.down_h_off
            w = w - op.left_w_off
            w = w - op.right_w_off
        if op.__class__.__name__ in ['ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            short_edge = int(round(short_edge * op.long_size / long_edge))
            long_edge = op.long_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['ResizeByShort']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            long_edge = int(round(long_edge * op.short_size / short_edge))
            short_edge = op.short_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['PaddingByAspectRatio']:
            reverse_list.append(('padding', (h, w)))
            ratio = w / h
            if ratio == op.aspect_ratio:
                pass
            elif ratio > op.aspect_ratio:
                h = int(w / op.aspect_ratio)
            else:
                w = int(h * op.aspect_ratio)
        if op.__class__.__name__ in ['LimitLong']:
            long_edge = max(h, w)
            short_edge = min(h, w)
            if ((op.max_long is not None) and (long_edge > op.max_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.max_long
                short_edge = int(round(short_edge * op.max_long / long_edge))
            elif ((op.min_long is not None) and (long_edge < op.min_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.min_long
                short_edge = int(round(short_edge * op.min_long / long_edge))
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
    return reverse_list


def reverse_transform(pred: paddle.Tensor, ori_shape: list, transforms: Callable, mode: str = 'nearest') -> paddle.Tensor:
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            # if paddle.get_device() == 'cpu':
            #     pred = paddle.cast(pred, 'uint8')
            #     pred = F.interpolate(pred, (h, w), mode=mode)
            #     pred = paddle.cast(pred, 'int32')
            # else:
            pred = F.interpolate(pred, (h, w), mode=mode)
        elif item[0] == 'crop':
            up_h_off, down_h_off = item[1][0], item[1][1]
            left_w_off, right_w_off = item[2][0], item[2][1]
            pred = F.pad(
                pred, [left_w_off, right_w_off, up_h_off, down_h_off],
                value=0,
                mode='constant',
                data_format="NCHW")
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


class Crop:
    """
    crop an image from four forwards.

    Args:
        up_h_off (int, optional): The cut height for image from up to down. Default: 0.
        down_h_off (int, optional): The cut height for image from down to up . Default: 0.
        left_w_off (int, optional): The cut height for image from left to right. Default: 0.
        right_w_off (int, optional): The cut width for image from right to left. Default: 0.
    """

    def __init__(self, up_h_off: int = 0, down_h_off: int = 0, left_w_off: int = 0, right_w_off: int = 0):
        self.up_h_off = up_h_off
        self.down_h_off = down_h_off
        self.left_w_off = left_w_off
        self.right_w_off = right_w_off

    def __call__(self, im: np.ndarray, label: np.ndarray = None) -> Tuple[np.ndarray]:
        if self.up_h_off < 0 or self.down_h_off < 0 or self.left_w_off < 0 or self.right_w_off < 0:
            raise Exception(
                "up_h_off, down_h_off, left_w_off,  right_w_off must equal or greater zero"
            )

        if self.up_h_off > 0 and self.up_h_off < im.shape[0]:
            im = im[self.up_h_off:, :, :]
            if label is not None:
                label = label[self.up_h_off:, :]

        if self.down_h_off > 0 and self.down_h_off < im.shape[0]:
            im = im[:-self.down_h_off, :, :]
            if label is not None:
                label = label[:-self.down_h_off, :]

        if self.left_w_off > 0 and self.left_w_off < im.shape[1]:
            im = im[:, self.left_w_off:, :]
            if label is not None:
                label = label[:, self.left_w_off:]

        if self.right_w_off > 0 and self.right_w_off < im.shape[1]:
            im = im[:, :-self.right_w_off, :]
            if label is not None:
                label = label[:, :-self.right_w_off]

        if label is None:
            return (im, )
        else:
            return (im, label)

def cv2_to_base64(image: np.ndarray) -> str:
    """
    Convert data from BGR to base64 format.
    """
    data = cv2.imencode('.png', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str: str) -> np.ndarray:
    """
    Convert data from base64 to BGR format.
    """
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data
