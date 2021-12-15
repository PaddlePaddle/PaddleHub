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
import time
import argparse
import os
from typing import Union, List, Tuple

import cv2
import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np
from paddlehub.module.module import moduleinfo, runnable, serving
import paddleseg.transforms as T
from paddleseg.utils import logger, progbar, visualize
from paddlehub.module.cv_module import ImageSegmentationModule
import paddleseg.utils as utils
from paddleseg.models import layers
from paddleseg.models import BiSeNetV2

from bisenet_lane_segmentation.processor import Crop, reverse_transform, cv2_to_base64, base64_to_cv2
from bisenet_lane_segmentation.lane_processor.tusimple_processor import TusimpleProcessor

@moduleinfo(
    name="bisenet_lane_segmentation",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="BiSeNetLane is a lane segmentation model.",
    version="1.0.0")
class BiSeNetLane(nn.Layer):
    """
    The BiSeNetLane use BiseNet V2 to process lane segmentation .

    Args:
        num_classes (int): The unique number of target classes.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes: int = 7,
                 lambd: float = 0.25,
                 align_corners: bool = False,
                 pretrained: str = None):
        super(BiSeNetLane, self).__init__()

        self.net = BiSeNetV2(
            num_classes=num_classes,
            lambd=lambd,
            align_corners=align_corners,
            pretrained=None)

        self.transforms = [Crop(up_h_off=160), T.Resize([640, 368]), T.Normalize()]
        self.cut_height = 160
        self.postprocessor = TusimpleProcessor(num_classes=7, cut_height=160,)

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'model.pdparams')
            model_dict = paddle.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")
            

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        logit_list = self.net(x)
        return logit_list
    
    def predict(self, image_list: list, visualization: bool = False, save_path: str = "bisenet_lane_segmentation_output") -> List[np.ndarray]:
        self.eval()
        result = []
        with paddle.no_grad():
            for i, im in enumerate(image_list):
                if isinstance(im, str):
                    im = cv2.imread(im)

                ori_shape = im.shape[:2]
                for op in self.transforms:
                    outputs = op(im)
                    im = outputs[0]

                im = np.transpose(im, (2, 0, 1))
                im = im[np.newaxis, ...]
                im = paddle.to_tensor(im)
                logit = self.forward(im)[0]
                pred = reverse_transform(logit, ori_shape, self.transforms, mode='bilinear')
                pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
                pred = paddle.squeeze(pred[0])
                pred = pred.numpy().astype('uint8')
                if visualization:
                    color_map = visualize.get_color_map_list(256)
                    pred_mask = visualize.get_pseudo_color_map(pred, color_map)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img_name = str(time.time()) + '.png'
                    image_save_path = os.path.join(save_path, img_name)
                    pred_mask.save(image_save_path)
                result.append(pred)
        return result

    @serving
    def serving_method(self, images: str, **kwargs) -> dict:
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        outputs = self.predict(image_list=images_decode, **kwargs)
        serving_data = [cv2_to_base64(outputs[i]) for i in range(len(outputs))]
        results = {'data': serving_data}

        return results

    @runnable
    def run_cmd(self, argvs: list) -> List[np.ndarray]:
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
            usage='%(prog)s',
            add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)

        results = self.predict(image_list=[args.input_path], save_path=args.output_dir, visualization=args.visualization)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default="bisenet_lane_segmentation_output", help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
        