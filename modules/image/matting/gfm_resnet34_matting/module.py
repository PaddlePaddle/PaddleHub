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
import time
import argparse
from typing import Callable, Union, List, Tuple

from PIL import Image
import numpy as np
import cv2
import scipy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlehub.module.module import moduleinfo
import paddlehub.vision.transforms as T
from paddlehub.module.module import moduleinfo, runnable, serving
from skimage.transform import resize

from gfm_resnet34_matting.gfm import GFM
import  gfm_resnet34_matting.processor as P 


@moduleinfo(
    name="gfm_resnet34_matting",
    type="CV/matting",
    author="paddlepaddle",
    author_email="",
    summary="gfm_resnet34_matting is an animal matting model.",
    version="1.0.0")
class GFMResNet34(nn.Layer):
    """
    The GFM implementation based on PaddlePaddle.
    
    The original article refers to：
    Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
    Main network file (GFM).
    
    Github repo: https://github.com/JizhiziLi/GFM
    Paper link (Arxiv): https://arxiv.org/abs/2010.16188
    """
    
    def __init__(self, pretrained: str=None):
        super(GFMResNet34, self).__init__()

        self.model = GFM()
        self.resize_by_short = P.ResizeByShort(1080)

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.model.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'model.pdparams')
            model_dict = paddle.load(checkpoint)
            self.model.set_dict(model_dict)
            print("load pretrained parameters success")

    def preprocess(self, img: Union[str, np.ndarray],  h: int, w: int) -> paddle.Tensor:
        if min(h, w) > 1080:
            img = self.resize_by_short(img)
        tensor_img = self.scale_image(img, h, w)
        return tensor_img
    
    def scale_image(self, img: np.ndarray, h: int, w: int, ratio: float = 1/3):
        new_h = min(1600, h - (h % 32))
        new_w = min(1600, w - (w % 32))
        resize_h = int(h*ratio)
        resize_w = int(w*ratio)
        new_h = min(1600, resize_h - (resize_h % 32))
        new_w = min(1600, resize_w - (resize_w % 32))

        scale_img = resize(img,(new_h,new_w)) * 255
        tensor_img = paddle.to_tensor(scale_img.astype(np.float32)[np.newaxis, :, :, :])
        tensor_img = tensor_img.transpose([0,3,1,2])
        return tensor_img


    def inference_img_scale(self, input: paddle.Tensor) -> List[paddle.Tensor]:
        pred_global, pred_local, pred_fusion = self.model(input)
        pred_global = P.gen_trimap_from_segmap_e2e(pred_global)
        pred_local = pred_local.numpy()[0,0,:,:]
        pred_fusion = pred_fusion.numpy()[0,0,:,:]
        return pred_global, pred_local, pred_fusion

    
    def predict(self, image_list: list, visualization: bool =True, save_path: str = "gfm_resnet34_matting_output"):
        self.model.eval()
        result = []
        with paddle.no_grad():
            for i, img in enumerate(image_list):
                if isinstance(img, str):
                    img = np.array(Image.open(img))[:,:,:3]
                else:
                    img = img[:,:,::-1]
                h, w, _ = img.shape
                tensor_img = self.preprocess(img, h, w)
                pred_glance_1, pred_focus_1, pred_fusion_1 = self.inference_img_scale(tensor_img)
                pred_glance_1 = resize(pred_glance_1,(h,w)) * 255.0
                tensor_img = self.scale_image(img, h, w, 1/2)
                pred_glance_2, pred_focus_2, pred_fusion_2 = self.inference_img_scale(tensor_img)
                pred_focus_2 = resize(pred_focus_2,(h,w))
                pred_fusion = P.get_masked_local_from_global_test(pred_glance_1, pred_focus_2)
                pred_fusion = (pred_fusion * 255).astype(np.uint8) 
                if visualization:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img_name = str(time.time()) + '.png'
                    image_save_path = os.path.join(save_path, img_name)
                    cv2.imwrite(image_save_path, pred_fusion)
                result.append(pred_fusion)
        return result

    @serving
    def serving_method(self, images: str, **kwargs):
        """
        Run as a service.
        """
        images_decode = [P.base64_to_cv2(image) for image in images]
        outputs = self.predict(image_list=images_decode, **kwargs)
        serving_data = [P.cv2_to_base64(outputs[i]) for i in range(len(outputs))]
        results = {'data': serving_data}

        return results

    @runnable
    def run_cmd(self, argvs: list):
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
            '--output_dir', type=str, default="gfm_resnet34_matting_output", help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
        
