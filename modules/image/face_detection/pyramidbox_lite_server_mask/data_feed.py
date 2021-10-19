# coding=utf-8
import os
import math
import time
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['reader']

multi_scales = [0.3, 0.6, 0.9]


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # nms
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def crop(image,
         pts,
         shift=0,
         scale=1.5,
         rotate=0,
         res_width=128,
         res_height=128):
    res = (res_width, res_height)
    idx1 = 0
    idx2 = 1
    # angle
    alpha = 0
    if pts[idx2, 0] != -1 and pts[idx2, 1] != -1 and pts[idx1, 0] != -1 and pts[
            idx1, 1] != -1:
        alpha = math.atan2(pts[idx2, 1] - pts[idx1, 1],
                           pts[idx2, 0] - pts[idx1, 0]) * 180 / math.pi
    pts[pts == -1] = np.inf
    coord_min = np.min(pts, 0)
    pts[pts == np.inf] = -1
    coord_max = np.max(pts, 0)
    # coordinates of center point
    c = np.array([
        coord_max[0] - (coord_max[0] - coord_min[0]) / 2,
        coord_max[1] - (coord_max[1] - coord_min[1]) / 2
    ])  # center
    max_wh = max((coord_max[0] - coord_min[0]) / 2,
                 (coord_max[1] - coord_min[1]) / 2)
    # Shift the center point, rot add eyes angle
    c = c + shift * max_wh
    rotate = rotate + alpha
    M = cv2.getRotationMatrix2D((c[0], c[1]), rotate,
                                res[0] / (2 * max_wh * scale))
    M[0, 2] = M[0, 2] - (c[0] - res[0] / 2.0)
    M[1, 2] = M[1, 2] - (c[1] - res[0] / 2.0)
    image_out = cv2.warpAffine(image, M, res)
    return image_out, M


def color_normalize(image, mean, std=None):
    if image.shape[-1] == 1:
        image = np.repeat(image, axis=2)
    h, w, c = image.shape
    image = np.transpose(image, (2, 0, 1))
    image = np.subtract(image.reshape(c, -1), mean[:, np.newaxis]).reshape(
        -1, h, w)
    image = np.transpose(image, (1, 2, 0))
    return image


def process_image(org_im, face):
    pts = np.array([
        face['left'], face['top'], face['right'], face['top'], face['left'],
        face['bottom'], face['right'], face['bottom']
    ]).reshape(4, 2).astype(np.float32)
    image_in, M = crop(org_im, pts)
    image_in = image_in / 256.0
    image_in = color_normalize(image_in, mean=np.array([0.5, 0.5, 0.5]))
    image_in = image_in.astype(np.float32).transpose([2, 0, 1]).reshape(
        -1, 3, 128, 128)
    return image_in


def reader(face_detector, shrink, confs_threshold, images, paths, use_gpu,
           use_multi_scale):
    """
    Preprocess to yield image.

    Args:
        face_detector (class): class to detect faces.
        shrink (float): parameter to control the resize scale in face_detector.
        confs_threshold (float): confidence threshold of face_detector.
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C], color space is BGR.
        paths (list[str]): paths to images.
        use_gpu (bool): whether to use gpu in face_detector.
        use_multi_scale (bool): whether to enable multi-scale face detection.
    Yield:
        element (collections.OrderedDict): info of original image, preprocessed image, contains 3 keys:
            org_im (numpy.ndarray) : original image.
            org_im_path (str): path to original image.
            preprocessed (list[OrderedDict]):each element contains 2 keys:
               face  (dict): face detected in the original image.
               image (numpy.ndarray): data to be fed into neural network.
    """
    component = list()
    if paths is not None:
        assert type(paths) is list, "paths should be a list."
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(
                im_path), "The {} isn't a valid file path.".format(im_path)
            im = cv2.imread(im_path)
            each['org_im'] = im
            each['org_im_path'] = im_path
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}'.format(
                round(time.time(), 6) * 1e6)
            component.append(each)

    for element in component:
        if use_multi_scale:
            scale_res = list()
            detect_faces = list()
            for scale in multi_scales:
                _detect_res = face_detector.face_detection(
                    images=[element['org_im']],
                    use_gpu=use_gpu,
                    visualization=False,
                    shrink=scale,
                    confs_threshold=confs_threshold)

                _s = list()
                for _face in _detect_res[0]['data']:
                    _face_list = [
                        _face['left'], _face['top'], _face['right'],
                        _face['bottom'], _face['confidence']
                    ]
                    _s.append(_face_list)

                if _s:
                    scale_res.append(np.array(_s))
            if scale_res:
            	scale_res = np.row_stack(scale_res)
            	scale_res = bbox_vote(scale_res)
            	keep_index = np.where(scale_res[:, 4] >= confs_threshold)[0]
            	scale_res = scale_res[keep_index, :]
            	for data in scale_res:
                    face = {
                    'left': data[0],
                    'top': data[1],
                    'right': data[2],
                    'bottom': data[3],
                    'confidence': data[4]
                    }
                    detect_faces.append(face)
            else:
                detect_faces = []
        else:
            _detect_res = face_detector.face_detection(
                images=[element['org_im']],
                use_gpu=use_gpu,
                visualization=False,
                shrink=shrink,
                confs_threshold=confs_threshold)
            detect_faces = _detect_res[0]['data']

        element['preprocessed'] = list()
        for face in detect_faces:
            handled = OrderedDict()
            handled['face'] = face
            handled['image'] = process_image(element['org_im'], face)
            element['preprocessed'].append(handled)

        yield element
