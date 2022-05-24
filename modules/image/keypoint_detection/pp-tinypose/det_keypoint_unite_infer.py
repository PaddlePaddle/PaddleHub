# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import json
import math
import os

import cv2
import numpy as np
import paddle
import yaml
from infer import Detector
from infer import PredictConfig
from keypoint_infer import KeyPointDetector
from keypoint_infer import PredictConfig_KeyPoint
from keypoint_postprocess import translate_to_ori_images
from preprocess import decode_image
from visualize import visualize_pose

KEYPOINT_SUPPORT_MODELS = {'HigherHRNet': 'keypoint_bottomup', 'HRNet': 'keypoint_topdown'}


def predict_with_given_det(image, det_res, keypoint_detector, keypoint_batch_size, run_benchmark):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(image, det_res)
    keypoint_vector = []
    score_vector = []

    rect_vector = det_rects
    keypoint_results = keypoint_detector.predict_image(rec_images, run_benchmark, repeats=10, visual=False)
    keypoint_vector, score_vector = translate_to_ori_images(keypoint_results, np.array(records))
    keypoint_res = {}
    keypoint_res['keypoint'] = [keypoint_vector.tolist(), score_vector.tolist()] if len(keypoint_vector) > 0 else [[],
                                                                                                                   []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res
