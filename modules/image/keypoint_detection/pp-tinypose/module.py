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
import argparse
import json
import math
import os
import time
from typing import Union

import cv2
import numpy as np
import paddle
import yaml
from det_keypoint_unite_infer import predict_with_given_det
from infer import Detector
from keypoint_infer import KeyPointDetector
from preprocess import base64_to_cv2
from preprocess import decode_image
from visualize import visualize_pose

from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(
    name="pp-tinypose",
    type="CV/keypoint_detection",
    author="paddlepaddle",
    author_email="",
    summary="PP-TinyPose is a real-time keypoint detection model optimized by PaddleDetecion for mobile devices.",
    version="1.0.0")
class PP_TinyPose:
    """
    PP-TinyPose Model.
    """

    def __init__(self):
        self.det_model_dir = os.path.join(self.directory, 'model/picodet_s_320_coco_lcnet/')
        self.keypoint_model_dir = os.path.join(self.directory, 'model/tinypose_256x192/')
        self.detector = Detector(self.det_model_dir)
        self.topdown_keypoint_detector = KeyPointDetector(self.keypoint_model_dir)

    def predict(self,
                img: Union[str, np.ndarray],
                save_path: str = "pp_tinypose_output",
                visualization: bool = False,
                use_gpu=False):
        if use_gpu:
            device = 'GPU'
        else:
            device = 'CPU'
        if self.detector.device != device:
            self.detector = Detector(self.det_model_dir, device=device)
            self.topdown_keypoint_detector = KeyPointDetector(self.keypoint_model_dir, device=device)

        self.visualization = visualization
        store_res = []

        # Decode image in advance in det + pose prediction
        image, _ = decode_image(img, {})
        results = self.detector.predict_image([image], visual=False)
        results = self.detector.filter_box(results, 0.5)
        if results['boxes_num'] > 0:
            keypoint_res = predict_with_given_det(image, results, self.topdown_keypoint_detector, 1, False)
            save_name = img if isinstance(img, str) else (str(time.time()) + '.png')
            store_res.append(
                [save_name, keypoint_res['bbox'], [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if self.visualization:
                visualize_pose(save_name, keypoint_res, visual_thresh=0.5, save_dir=save_path)
        return store_res

    @serving
    def serving_method(self, images: list, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(img=images_decode[0], **kwargs)
        results = json.dumps(results)
        return results

    @runnable
    def run_cmd(self, argvs: list):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.predict(img=args.input_path,
                               save_path=args.output_dir,
                               visualization=args.visualization,
                               use_gpu=args.use_gpu)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='pp_tinypose_output',
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=bool,
                                           default=True,
                                           help="whether to save output as images.")

        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
