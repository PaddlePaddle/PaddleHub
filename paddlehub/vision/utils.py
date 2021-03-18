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

import os
from typing import Callable, Union, List, Tuple

import cv2
import paddle
import PIL
import numpy as np
import matplotlib as plt
import paddle.nn.functional as F


def is_image_file(filename: str) -> bool:
    '''Determine whether the input file name is a valid image file name.'''
    ext = os.path.splitext(filename)[-1].lower()
    return ext in ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff']


def get_img_file(dir_name: str) -> List[str]:
    '''Get all image file paths in several directories which have the same parent directory.'''
    images = []
    for parent, _, filenames in os.walk(dir_name):
        for filename in filenames:
            if not is_image_file(filename):
                continue
            img_path = os.path.join(parent, filename)
            images.append(img_path)

    return images


def box_crop(boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, crop: List[int], img_shape: List[int]) -> Tuple:
    """Crop the boxes ,labels, scores according to the given shape"""

    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    scores = scores * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, scores, mask.sum()


def box_iou_xywh(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate iou by xywh"""

    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def draw_boxes_on_image(image_path: str,
                        boxes: np.ndarray,
                        scores: np.ndarray,
                        labels: np.ndarray,
                        label_names: List[str],
                        score_thresh: float = 0.5,
                        save_path: str = 'result'):
    """Draw boxes on images."""
    image = np.array(PIL.Image.open(image_path))
    plt.figure()
    _, ax = plt.subplots(1)
    ax.imshow(image)

    image_name = image_path.split('/')[-1]
    print("Image {} detect: ".format(image_name))
    colors = {}
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = int(label)
        if label not in colors:
            colors[label] = plt.get_cmap('hsv')(label / len(label_names))
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor=colors[label])
        ax.add_patch(rect)
        ax.text(
            x1,
            y1,
            '{} {:.4f}'.format(label_names[label], score),
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox={
                'facecolor': colors[label],
                'alpha': 0.5,
                'pad': 0
            },
            fontsize=8,
            color='white')
        print("\t {:15s} at {:25} score: {:.5f}".format(label_names[int(label)], str(list(map(int, list(box)))), score))
    image_name = image_name.replace('jpg', 'png')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("{}/{}".format(save_path, image_name), bbox_inches='tight', pad_inches=0.0)
    plt.cla()
    plt.close('all')


def get_label_infos(file_list: str) -> str:
    """Get label names by corresponding category ids."""
    from pycocotools.coco import COCO
    map_label = COCO(file_list)
    label_names = []
    categories = map_label.loadCats(map_label.getCatIds())
    for category in categories:
        label_names.append(category['name'])
    return label_names


def subtract_imagenet_mean_batch(batch: paddle.Tensor) -> paddle.Tensor:
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    mean = np.zeros(shape=batch.shape, dtype='float32')
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    mean = paddle.to_tensor(mean)
    return batch - mean


def gram_matrix(data: paddle.Tensor) -> paddle.Tensor:
    """Get gram matrix"""
    b, ch, h, w = data.shape
    features = data.reshape((b, ch, w * h))
    features_t = features.transpose((0, 2, 1))
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def npmax(array: np.ndarray) -> Tuple[int]:
    """Get max value and index."""
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def visualize(image:  Union[np.ndarray, str], result: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """
    Convert segmentation result to color image, and save added image.

    Args:
        image (str|np.ndarray): The path of origin image or bgr image.
        result (np.ndarray): The predict result of image.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): return the visualized result.
    """

    color_map = get_color_map_list(256)
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))
    if isinstance(image, str):
        im = cv2.imread(image)
    else:
        im = image
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    return vis_result


def get_pseudo_color_map(pred: np.ndarray) -> PIL.Image.Image:
    '''visualization the segmentation mask.'''
    pred_mask = PIL.Image.fromarray(pred.astype(np.uint8), mode='P')
    color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes: int) -> List[int]:
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def get_reverse_list(ori_shape: List[int], transforms: List[Callable]) -> List[tuple]:
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
        if op.__class__.__name__ in ['Resize', 'ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
    return reverse_list


def reverse_transform(pred: paddle.Tensor, ori_shape: List[int], transforms: List[int]) -> paddle.Tensor:
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            pred = F.interpolate(pred, (h, w), mode='nearest')
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred