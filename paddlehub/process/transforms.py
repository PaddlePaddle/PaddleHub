# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import random
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from paddlehub.process.functional import *


class Compose:
    def __init__(self, transforms, to_rgb=True, stay_rgb=False, is_permute=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                             'must be equal or larger than 1!')
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.stay_rgb = stay_rgb
        self.is_permute = is_permute

    def __call__(self, im):
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')

        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            im = op(im)

        if not self.stay_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        if self.is_permute:
            im = permute(im)

        return im


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im):
        if random.random() < self.prob:
            im = horizontal_flip(im)
        return im


class RandomVerticalFlip:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im):
        if random.random() < self.prob:
            im = vertical_flip(im)
        return im


class Resize:
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=512, interp='LINEAR'):
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

    def __call__(self, im):
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        return im


class ResizeByLong:
    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, im):
        im = resize_long(im, self.long_size)
        return im


class ResizeRangeScaling:
    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(min_value, max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, im):
        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(np.random.uniform(self.min_value, self.max_value) + 0.5)
        im = resize_long(im, random_size, cv2.INTER_LINEAR)
        return im


class ResizeStepScaling:
    def __init__(self, min_scale_factor=0.75, max_scale_factor=1.25, scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError('min_scale_factor must be less than max_scale_factor, '
                             'but they are {} and {}.'.format(min_scale_factor, max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, im):
        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor, self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) / self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor, self.max_scale_factor, num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * im.shape[1]))
        h = int(round(scale_factor * im.shape[0]))

        im = resize(im, (w, h), cv2.INTER_LINEAR)
        return im


class Normalize:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise ValueError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)
        return im


class Padding:
    def __init__(self, target_size, im_padding_value=[127.5, 127.5, 127.5]):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'.format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError("Type of target_size is invalid. Must be Integer or List or tuple, now is {}".format(
                type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value

    def __call__(self, im):
        im_height, im_width = im.shape[0], im.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'the size of image should be less than target_size, but the size of image ({}, {}), is larger than target_size ({}, {})'
                .format(im_width, im_height, target_width, target_height))
        else:
            im = cv2.copyMakeBorder(im, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=self.im_padding_value)

        return im


class RandomPaddingCrop:
    def __init__(self, crop_size=512, im_padding_value=[127.5, 127.5, 127.5]):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'when crop_size is list or tuple, it should include 2 elements, but it is {}'.format(crop_size))
        elif not isinstance(crop_size, int):
            raise TypeError("Type of crop_size is invalid. Must be Integer or List or tuple, now is {}".format(
                type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value

    def __call__(self, im):
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if img_height == crop_height and img_width == crop_width:
            return im
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im = cv2.copyMakeBorder(im,
                                        0,
                                        pad_height,
                                        0,
                                        pad_width,
                                        cv2.BORDER_CONSTANT,
                                        value=self.im_padding_value)

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im = im[h_off:(crop_height + h_off), w_off:(w_off + crop_width), :]

            return im


class RandomBlur:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)

        return im


class RandomRotation:
    def __init__(self, max_rotation=15, im_padding_value=[127.5, 127.5, 127.5]):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value

    def __call__(self, im):
        if self.max_rotation > 0:
            (h, w) = im.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation, self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            im = cv2.warpAffine(im,
                                r,
                                dsize=dsize,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=self.im_padding_value)

        return im


class RandomScaleAspect:
    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im):
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio, 1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    im = cv2.resize(im, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        return im


class RandomDistort:
    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob

    def __call__(self, im):
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        ops = [brightness, contrast, saturation, hue]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob
        }
        im = im.astype('uint8')
        im = Image.fromarray(im)
        for id in range(4):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        im = np.asarray(im).astype('float32')

        return im


class ConvertColorSpace:
    """
    Convert color space from RGB to LAB or from LAB to RGB.

    Args:
       mode(str): Color space convert mode, it can be 'RGB2LAB' or 'LAB2RGB'.

    Return:
        img(np.ndarray): converted image.
    """
    def __init__(self, mode: str = 'RGB2LAB'):
        self.mode = mode

    def rgb2xyz(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert color space from RGB to XYZ.

        Args:
           img(np.ndarray): Original RGB image.

        Return:
            img(np.ndarray): Converted XYZ image.
        """
        mask = (rgb > 0.04045)
        np.seterr(invalid='ignore')
        rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)
        rgb = np.nan_to_num(rgb)
        x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
        y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
        z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
        out = np.concatenate((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), axis=1)
        return out

    def xyz2lab(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert color space from XYZ to LAB.

        Args:
           img(np.ndarray): Original XYZ image.

        Return:
            img(np.ndarray): Converted LAB image.
        """
        sc = np.array((0.95047, 1., 1.08883))[None, :, None, None]
        xyz_scale = xyz / sc
        mask = (xyz_scale > .008856).astype(np.float32)
        xyz_int = np.cbrt(xyz_scale) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)
        L = 116. * xyz_int[:, 1, :, :] - 16.
        a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
        b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
        out = np.concatenate((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), axis=1)
        return out

    def rgb2lab(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert color space from RGB to LAB.

        Args:
           img(np.ndarray): Original RGB image.

        Return:
            img(np.ndarray): Converted LAB image.
        """
        lab = self.xyz2lab(self.rgb2xyz(rgb))
        l_rs = (lab[:, [0], :, :] - 50) / 100
        ab_rs = lab[:, 1:, :, :] / 110
        out = np.concatenate((l_rs, ab_rs), axis=1)
        return out

    def xyz2rgb(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert color space from XYZ to RGB.

        Args:
           img(np.ndarray): Original XYZ image.

        Return:
            img(np.ndarray): Converted RGB image.
        """
        r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
        g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
        b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]
        rgb = np.concatenate((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), axis=1)
        rgb = np.maximum(rgb, 0)  # sometimes reaches a small negative number, which causes NaNs
        mask = (rgb > .0031308).astype(np.float32)
        np.seterr(invalid='ignore')
        out = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
        out = np.nan_to_num(out)
        return out

    def lab2xyz(self, lab: np.ndarray) -> np.ndarray:
        """
        Convert color space from LAB to XYZ.

        Args:
           img(np.ndarray): Original LAB image.

        Return:
            img(np.ndarray): Converted XYZ image.
        """
        y_int = (lab[:, 0, :, :] + 16.) / 116.
        x_int = (lab[:, 1, :, :] / 500.) + y_int
        z_int = y_int - (lab[:, 2, :, :] / 200.)
        z_int = np.maximum(z_int, 0)
        out = np.concatenate((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), axis=1)
        mask = (out > .2068966).astype(np.float32)
        np.seterr(invalid='ignore')
        out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)
        out = np.nan_to_num(out)
        sc = np.array((0.95047, 1., 1.08883))[None, :, None, None]
        out = out * sc
        return out

    def lab2rgb(self, lab_rs: np.ndarray) -> np.ndarray:
        """
        Convert color space from LAB to RGB.

        Args:
           img(np.ndarray): Original LAB image.

        Return:
            img(np.ndarray): Converted RGB image.
        """
        l = lab_rs[:, [0], :, :] * 100 + 50
        ab = lab_rs[:, 1:, :, :] * 110
        lab = np.concatenate((l, ab), axis=1)
        out = self.xyz2rgb(self.lab2xyz(lab))
        return out

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.mode == 'RGB2LAB':
            img = np.expand_dims(img / 255, 0)
            img = np.array(img).transpose(0, 3, 1, 2)
            return self.rgb2lab(img)
        elif self.mode == 'LAB2RGB':
            return self.lab2rgb(img)
        else:
            raise ValueError('The mode should be RGB2LAB or LAB2RGB')


class ColorizeHint:
    """Get hint and mask images for colorization.

    This method is prepared for user guided colorization tasks. Take the original RGB images as imput, we will obtain the local hints and correspoding mask to guid colorization process.

    Args:
       percent(float): Probability for ignoring hint in an iteration.
       num_points(int): Number of selected hints in an iteration.
       samp(str): Sample method, default is normal.
       use_avg(bool): Whether to use mean in selected hint area.

    Return:
        hint(np.ndarray): hint images
        mask(np.ndarray): mask images
    """
    def __init__(self, percent: float, num_points: int = None, samp: str = 'normal', use_avg: bool = True):
        self.percent = percent
        self.num_points = num_points
        self.samp = samp
        self.use_avg = use_avg

    def __call__(self, data: np.ndarray, hint: np.ndarray, mask: np.ndarray):
        sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data = data
        self.hint = hint
        self.mask = mask
        N, C, H, W = data.shape
        for nn in range(N):
            pp = 0
            cont_cond = True
            while cont_cond:
                if self.num_points is None:  # draw from geometric
                    # embed()
                    cont_cond = np.random.rand() > (1 - self.percent)
                else:  # add certain number of points
                    cont_cond = pp < self.num_points
                if not cont_cond:  # skip out of loop if condition not met
                    continue
                P = np.random.choice(sample_Ps)  # patch size
                # sample location
                if self.samp == 'normal':  # geometric distribution
                    h = int(np.clip(np.random.normal((H - P + 1) / 2., (H - P + 1) / 4.), 0, H - P))
                    w = int(np.clip(np.random.normal((W - P + 1) / 2., (W - P + 1) / 4.), 0, W - P))
                else:  # uniform distribution
                    h = np.random.randint(H - P + 1)
                    w = np.random.randint(W - P + 1)
                # add color point
                if self.use_avg:
                    # embed()
                    hint[nn, :, h:h + P, w:w + P] = np.mean(np.mean(data[nn, :, h:h + P, w:w + P],
                                                                    axis=2,
                                                                    keepdims=True),
                                                            axis=1,
                                                            keepdims=True).reshape(1, C, 1, 1)
                else:
                    hint[nn, :, h:h + P, w:w + P] = data[nn, :, h:h + P, w:w + P]
                mask[nn, :, h:h + P, w:w + P] = 1
                # increment counter
                pp += 1

        mask -= 0.5
        return hint, mask


class SqueezeAxis:
    """
    Squeeze the specific axis when it equal to 1.

    Args:
       axis(int): Which axis should be squeezed.

    """
    def __init__(self, axis: int):
        self.axis = axis

    def __call__(self, data: dict):
        if isinstance(data, dict):
            for key in data.keys():
                data[key] = np.squeeze(data[key], 0).astype(np.float32)
            return data
        else:
            raise TypeError("Type of data is invalid. Must be Dict or List or tuple, now is {}".format(type(data)))


class ColorizePreprocess:
    """Prepare dataset for image Colorization.

    Args:
       ab_thresh(float): Thresh value for setting mask value.
       p(float): Probability for ignoring hint in an iteration.
       num_points(int): Number of selected hints in an iteration.
       samp(str): Sample method, default is normal.
       use_avg(bool): Whether to use mean in selected hint area.
       is_train(bool): Training process or not.

    Return:
        data(dict)：The preprocessed data for colorization.

    """
    def __init__(self,
                 ab_thresh: float = 0.,
                 p: float = 0.,
                 num_points: int = None,
                 samp: str = 'normal',
                 use_avg: bool = True,
                 is_train: bool = True):
        self.ab_thresh = ab_thresh
        self.p = p
        self.num_points = num_points
        self.samp = samp
        self.use_avg = use_avg
        self.is_train = is_train
        self.gethint = ColorizeHint(percent=self.p, num_points=self.num_points, samp=self.samp, use_avg=self.use_avg)
        self.squeeze = SqueezeAxis(0)

    def __call__(self, data_lab: np.ndarray):
        """
        This method seperates the L channel and AB channel, obtain hint, mask and real_B_enc as the input for colorization task.

        Args:
           img(np.ndarray): LAB image.

        Returns:
            data(dict)：The preprocessed data for colorization.
        """
        data = {}
        A = 2 * 110 / 10 + 1
        data['A'] = data_lab[:, [
            0,
        ], :, :]
        data['B'] = data_lab[:, 1:, :, :]
        if self.ab_thresh > 0:  # mask out grayscale images
            thresh = 1. * self.ab_thresh / 110
            mask = np.sum(np.abs(np.max(np.max(data['B'], axis=3), axis=2) - np.min(np.min(data['B'], axis=3), axis=2)),
                          axis=1)
            mask = (mask >= thresh)
            data['A'] = data['A'][mask, :, :, :]
            data['B'] = data['B'][mask, :, :, :]
            if np.sum(mask) == 0:
                return None
        data_ab_rs = np.round((data['B'][:, :, ::4, ::4] * 110. + 110.) / 10.)  # normalized bin number
        data['real_B_enc'] = data_ab_rs[:, [0], :, :] * A + data_ab_rs[:, [1], :, :]
        data['hint_B'] = np.zeros(shape=data['B'].shape)
        data['mask_B'] = np.zeros(shape=data['A'].shape)
        data['hint_B'], data['mask_B'] = self.gethint(data['B'], data['hint_B'], data['mask_B'])
        if self.is_train:
            data = self.squeeze(data)
            data['real_B_enc'] = data['real_B_enc'].astype(np.int64)
        else:
            data['A'] = data['A'].astype(np.float32)
            data['B'] = data['B'].astype(np.float32)
            data['real_B_enc'] = data['real_B_enc'].astype(np.int64)
            data['hint_B'] = data['hint_B'].astype(np.float32)
            data['mask_B'] = data['mask_B'].astype(np.float32)
        return data


class ColorPostprocess:
    """
    Transform images from [0, 1] to [0, 255]

    Args:
       type(type): Type of Image value.

    Return:
        img(np.ndarray): Image in range of 0-255.
    """
    def __init__(self, type: type = np.uint8):
        self.type = type

    def __call__(self, img: np.ndarray):
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1) * 255
        img = img.astype(self.type)
        return img


class CenterCrop:
    """
        Crop the middle part of the image to the specified size.

        Args:
           crop_size(int): Crop size.

        Return:
            img(np.ndarray): Croped image.
    """
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, img: np.ndarray):
        img_width, img_height, chanel = img.shape
        crop_top = int((img_height - self.crop_size) / 2.)
        crop_left = int((img_width - self.crop_size) / 2.)
        return img[crop_left:crop_left + self.crop_size, crop_top:crop_top + self.crop_size, :]


class SetType:
    """
    Set image type.

    Args:
       type(type): Type of Image value.

    Return:
        img(np.ndarray): Transformed image.
    """
    def __init__(self, datatype: type = 'float32'):
        self.type = datatype

    def __call__(self, img: np.ndarray):
        img = img.astype(self.type)
        return img
