import copy
import os
import random
from typing import Callable

import cv2
import numpy as np
import matplotlib
import PIL
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

from paddlehub.process.functional import *

matplotlib.use('Agg')


class DetectCatagory:
    """Load label name, id and map from detection dataset.

    Args:
        attrbox(Callable): Method to get detection attributes of images.
        data_dir(str): Image dataset path.

    Returns:
        label_names(List(str)): The dataset label names.
        label_ids(List(int)): The dataset label ids.
        category_to_id_map(dict): Mapping relations of category and id for images.
    """
    def __init__(self, attrbox: Callable, data_dir: str):
        self.attrbox = attrbox
        self.img_dir = data_dir

    def __call__(self):
        self.categories = self.attrbox.loadCats(self.attrbox.getCatIds())
        self.num_category = len(self.categories)
        label_names = []
        label_ids = []
        for category in self.categories:
            label_names.append(category['name'])
            label_ids.append(int(category['id']))
        category_to_id_map = {v: i for i, v in enumerate(label_ids)}
        return label_names, label_ids, category_to_id_map


class ParseImages:
    """Prepare images for detection.

    Args:
        attrbox(Callable): Method to get detection attributes of images.
        is_train(bool): Select the mode for train or test.
        data_dir(str): Image dataset path.
        category_to_id_map(dict): Mapping relations of category and id for images.

    Returns:
        imgs(dict): The input for detection model, it is a dict.
    """
    def __init__(self, attrbox: Callable, data_dir: str, category_to_id_map: dict):
        self.attrbox = attrbox
        self.img_dir = data_dir
        self.category_to_id_map = category_to_id_map
        self.parse_gt_annotations = GTAnotations(self.attrbox, self.category_to_id_map)

    def __call__(self):
        image_ids = self.attrbox.getImgIds()
        image_ids.sort()
        imgs = copy.deepcopy(self.attrbox.loadImgs(image_ids))

        for img in imgs:
            img['image'] = os.path.join(self.img_dir, img['file_name'])
            assert os.path.exists(img['image']), "image {} not found.".format(img['image'])
            box_num = 50
            img['gt_boxes'] = np.zeros((box_num, 4), dtype=np.float32)
            img['gt_labels'] = np.zeros((box_num), dtype=np.int32)
            img = self.parse_gt_annotations(img)
        return imgs


class GTAnotations:
    """Set gt boxes and gt labels for train.

    Args:
        attrbox(Callable): Method for get detection attributes for images.
        category_to_id_map(dict): Mapping relations of category and id for images.
        img(dict): Input for detection model.

    Returns:
        img(dict): Set specific value on the attributes of 'gt boxes' and 'gt labels' for input.
    """
    def __init__(self, attrbox: Callable, category_to_id_map: dict):
        self.attrbox = attrbox
        self.category_to_id_map = category_to_id_map

    def __call__(self, img: dict):
        img_height = img['height']
        img_width = img['width']
        anno = self.attrbox.loadAnns(self.attrbox.getAnnIds(imgIds=img['id'], iscrowd=None))
        gt_index = 0

        for target in anno:
            if target['area'] < -1:
                continue
            if 'ignore' in target and target['ignore']:
                continue
            box = coco_anno_box_to_center_relative(target['bbox'], img_height, img_width)

            if box[2] <= 0 and box[3] <= 0:
                continue
            img['gt_boxes'][gt_index] = box
            img['gt_labels'][gt_index] = \
                self.category_to_id_map[target['category_id']]
            gt_index += 1
            if gt_index >= 50:
                break
        return img


class RandomDistort:
    """ Distort the input image randomly.
    Args:
        lower(float): The lower bound value for enhancement, default is 0.5.
        upper(float): The upper bound value for enhancement, default is 1.5.

    Returns:
        img(np.ndarray): Distorted image.
        data(dict): Image info and label info.

    """
    def __init__(self, lower: float = 0.5, upper: float = 1.5):
        self.lower = lower
        self.upper = upper

    def random_brightness(self, img: PIL.Image):
        e = np.random.uniform(self.lower, self.upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(self, img: PIL.Image):
        e = np.random.uniform(self.lower, self.upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(self, img: PIL.Image):
        e = np.random.uniform(self.lower, self.upper)
        return ImageEnhance.Color(img).enhance(e)

    def __call__(self, img: np.ndarray, data: dict):
        ops = [self.random_brightness, self.random_contrast, self.random_color]
        np.random.shuffle(ops)
        img = Image.fromarray(img)
        img = ops[0](img)
        img = ops[1](img)
        img = ops[2](img)
        img = np.asarray(img)

        return img, data


class RandomExpand:
    """Randomly expand images and gt boxes by random ratio. It is a data enhancement operation for model training.
    Args:
        max_ratio(float): Max value for expansion ratio, default is 4.
        fill(list): Initialize the pixel value of the image with the input fill value, default is None.
        keep_ratio(bool): Whether image keeps ratio.
        thresh(float): If random ratio does not exceed the thresh, return original images and gt boxes, default is 0.5.

    Return:
        img(np.ndarray): Distorted image.
        data(dict): Image info and label info.

    """
    def __init__(self, max_ratio: float = 4., fill: list = None, keep_ratio: bool = True, thresh: float = 0.5):

        self.max_ratio = max_ratio
        self.fill = fill
        self.keep_ratio = keep_ratio
        self.thresh = thresh

    def __call__(self, img: np.ndarray, data: dict):
        gtboxes = data['gt_boxes']

        if random.random() > self.thresh:
            return img, data
        if self.max_ratio < 1.0:
            return img, data
        h, w, c = img.shape

        ratio_x = random.uniform(1, self.max_ratio)
        if self.keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, self.max_ratio)

        oh = int(h * ratio_y)
        ow = int(w * ratio_x)
        off_x = random.randint(0, ow - w)
        off_y = random.randint(0, oh - h)

        out_img = np.zeros((oh, ow, c))
        if self.fill and len(self.fill) == c:
            for i in range(c):
                out_img[:, :, i] = self.fill[i] * 255.0

        out_img[off_y:off_y + h, off_x:off_x + w, :] = img
        gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
        gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
        gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
        gtboxes[:, 3] = gtboxes[:, 3] / ratio_y
        data['gt_boxes'] = gtboxes
        img = out_img.astype('uint8')

        return img, data


class RandomCrop:
    """
    Random crop the input image according to constraints.
    Args:
        scales(list): The value of the cutting area relative to the original area, expressed in the form of \
                      [min, max]. The default value is [.3, 1.].
        max_ratio(float): Max ratio of the original area relative to the cutting area, default is 2.0.
        constraints(list): The value of min and max iou values, default is None.
        max_trial(int): The max trial for finding a valid crop area. The default value is 50.

    Returns:
        img(np.ndarray): Distorted image.
        data(dict): Image info and label info.

    """
    def __init__(self,
                 scales: list = [0.3, 1.0],
                 max_ratio: float = 2.0,
                 constraints: list = None,
                 max_trial: int = 50):
        self.scales = scales
        self.max_ratio = max_ratio
        self.constraints = constraints
        self.max_trial = max_trial

    def __call__(self, img: np.ndarray, data: dict):
        boxes = data['gt_boxes']
        labels = data['gt_labels']
        scores = data['gt_scores']

        if len(boxes) == 0:
            return img, data
        if not self.constraints:
            self.constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.0, 1.0)]

        img = Image.fromarray(img)
        w, h = img.size
        crops = [(0, 0, w, h)]
        for min_iou, max_iou in self.constraints:
            for _ in range(self.max_trial):
                scale = random.uniform(self.scales[0], self.scales[1])
                aspect_ratio = random.uniform(max(1 / self.max_ratio, scale * scale), \
                                              min(self.max_ratio, 1 / scale / scale))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_x = random.randrange(w - crop_w)
                crop_y = random.randrange(h - crop_h)
                crop_box = np.array([[(crop_x + crop_w / 2.0) / w, (crop_y + crop_h / 2.0) / h, crop_w / float(w),
                                      crop_h / float(h)]])
                iou = box_iou_xywh(crop_box, boxes)
                if min_iou <= iou.min() and max_iou >= iou.max():
                    crops.append((crop_x, crop_y, crop_w, crop_h))
                    break

        while crops:
            crop = crops.pop(np.random.randint(0, len(crops)))
            crop_boxes, crop_labels, crop_scores, box_num = box_crop(boxes, labels, scores, crop, (w, h))

            if box_num < 1:
                continue
            img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
            img = np.asarray(img)
            data['gt_boxes'] = crop_boxes
            data['gt_labels'] = crop_labels
            data['gt_scores'] = crop_scores
            return img, data
        img = np.asarray(img)
        data['gt_boxes'] = boxes
        data['gt_labels'] = labels
        data['gt_scores'] = scores
        return img, data


class RandomFlip:
    """Flip the images and gt boxes randomly.
    Args:
        thresh: Probability for random flip.
    Returns:
        img(np.ndarray): Distorted image.
        data(dict): Image info and label info.
    """
    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh

    def __call__(self, img, data):
        gtboxes = data['gt_boxes']
        if random.random() > self.thresh:
            img = img[:, ::-1, :]
            gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
        data['gt_boxes'] = gtboxes
        return img, data


class Compose:
    """Preprocess the input data according to the operators.
    Args:
        transforms(list): Preprocessing operators.
    Returns:
        img(np.ndarray): Preprocessed image.
        data(dict): Image info and label info, default is None.
    """
    def __init__(self, transforms: list):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                             'must be equal or larger than 1!')
        self.transforms = transforms

    def __call__(self, data: dict):

        if isinstance(data, dict):
            if isinstance(data['image'], str):
                img = cv2.imread(data['image'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_labels = data['gt_labels'].copy()
            data['gt_scores'] = np.ones_like(gt_labels)
            for op in self.transforms:
                img, data = op(img, data)
            img = img.transpose((2, 0, 1))
            return img, data

        if isinstance(data, str):
            img = cv2.imread(data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for op in self.transforms:
                img, data = op(img, data)
            img = img.transpose((2, 0, 1))
            return img


class Resize:
    """Resize the input images.
    Args:
        target_size(int): Targeted input size.
        interp(str): Interpolation method.
    Returns:
        img(np.ndarray): Preprocessed image.
        data(dict): Image info and label info, default is None.
    """
    def __init__(self, target_size: int = 512, interp: str = 'RANDOM'):
        self.interp_dict = {
            'NEAREST': cv2.INTER_NEAREST,
            'LINEAR': cv2.INTER_LINEAR,
            'CUBIC': cv2.INTER_CUBIC,
            'AREA': cv2.INTER_AREA,
            'LANCZOS4': cv2.INTER_LANCZOS4
        }
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("interp should be one of {}".format(self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise TypeError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'.format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError("Type of target_size is invalid. Must be Integer or List or tuple, now is {}".format(
                type(target_size)))

        self.target_size = target_size

    def __call__(self, img, data=None):

        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        img = resize(img, self.target_size, self.interp_dict[interp])
        if data is not None:
            return img, data
        else:
            return img


class Normalize:
    """Normalize the input images.
    Args:
        mean(list): Mean values for normalization, default is [0.5, 0.5, 0.5].
        std(list): Standard deviation for normalization, default is [0.5, 0.5, 0.5].
    Returns:
        img(np.ndarray): Preprocessed image.
        data(dict): Image info and label info, default is None.
    """
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise ValueError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, data=None):
        if data is not None:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im = normalize(im, mean, std)
            return im, data
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im = normalize(im, mean, std)
            return im


class ShuffleBox:
    """Shuffle data information."""
    def __call__(self, img, data):
        gt = np.concatenate([data['gt_boxes'], data['gt_labels'][:, np.newaxis], data['gt_scores'][:, np.newaxis]],
                            axis=1)
        idx = np.arange(gt.shape[0])
        np.random.shuffle(idx)
        gt = gt[idx, :]
        data['gt_boxes'], data['gt_labels'], data['gt_scores'] = gt[:, :4], gt[:, 4], gt[:, 5]
        return img, data
