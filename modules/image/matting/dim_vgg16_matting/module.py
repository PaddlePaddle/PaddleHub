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

import numpy as np
import cv2
import scipy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlehub.module.module import moduleinfo
import paddlehub.vision.segmentation_transforms as T
from paddlehub.module.module import moduleinfo, runnable, serving
from paddleseg.models import layers

from dim_vgg16_matting.vgg import VGG16
import dim_vgg16_matting.processor as P


@moduleinfo(
    name="dim_vgg16_matting", 
    type="CV/matting", 
    author="paddlepaddle",
    summary="dim_vgg16_matting is a matting model",  
    version="1.0.0"  
)
class DIMVGG16(nn.Layer):
    """
    The DIM implementation based on PaddlePaddle.

    The original article refers to
    Ning Xu, et, al. "Deep Image Matting"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:
        stage (int, optional): The stage of model. Defautl: 3.
        decoder_input_channels(int, optional): The channel of decoder input. Default: 512.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.

    """
    def __init__(self,
                 stage: int = 3,
                 decoder_input_channels: int = 512,
                 pretrained: str = None):
        super(DIMVGG16, self).__init__()
        
        self.backbone = VGG16()
        self.pretrained = pretrained
        self.stage = stage

        decoder_output_channels = [64, 128, 256, 512]
        self.decoder = Decoder(
            input_channels=decoder_input_channels,
            output_channels=decoder_output_channels)
        if self.stage == 2:
            for param in self.backbone.parameters():
                param.stop_gradient = True
            for param in self.decoder.parameters():
                param.stop_gradient = True
        if self.stage >= 2:
            self.refine = Refine()
            
        self.transforms = P.Compose([P.LoadImages(), P.LimitLong(max_long=3840),P.Normalize()])

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'dim-vgg16.pdparams')
            model_dict = paddle.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")
    
    def preprocess(self, img: Union[str, np.ndarray] , transforms: Callable, trimap: Union[str, np.ndarray] = None) -> dict:
        data = {}
        data['img'] = img
        if trimap is not None:
            data['trimap'] = trimap
            data['gt_fields'] = ['trimap']
        data['trans_info'] = []
        data = self.transforms(data)
        data['img'] = paddle.to_tensor(data['img'])
        data['img'] = data['img'].unsqueeze(0)
        if trimap is not None:
            data['trimap'] = paddle.to_tensor(data['trimap'])
            data['trimap'] = data['trimap'].unsqueeze((0, 1))

        return data  
    
    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        input_shape = paddle.shape(inputs['img'])[-2:]
        x = paddle.concat([inputs['img'], inputs['trimap'] / 255], axis=1)
        fea_list = self.backbone(x)

        # decoder stage
        up_shape = []
        for i in range(5):
            up_shape.append(paddle.shape(fea_list[i])[-2:])
        alpha_raw = self.decoder(fea_list, up_shape)
        alpha_raw = F.interpolate(
            alpha_raw, input_shape, mode='bilinear', align_corners=False)
        logit_dict = {'alpha_raw': alpha_raw}
        if self.stage < 2:
            return logit_dict

        if self.stage >= 2:
            # refine stage
            refine_input = paddle.concat([inputs['img'], alpha_raw], axis=1)
            alpha_refine = self.refine(refine_input)

            # finally alpha
            alpha_pred = alpha_refine + alpha_raw
            alpha_pred = F.interpolate(
                alpha_pred, input_shape, mode='bilinear', align_corners=False)
            if not self.training:
                alpha_pred = paddle.clip(alpha_pred, min=0, max=1)
            logit_dict['alpha_pred'] = alpha_pred
   
        return alpha_pred
    
    def predict(self, image_list: list, trimap_list: list, visualization: bool =False, save_path: str = "dim_vgg16_matting_output") -> list:
        self.eval()
        result= []
        with paddle.no_grad():
            for i, im_path in enumerate(image_list):
                trimap = trimap_list[i] if trimap_list is not None else None
                data = self.preprocess(img=im_path, transforms=self.transforms, trimap=trimap)
                alpha_pred = self.forward(data)
                alpha_pred = P.reverse_transform(alpha_pred, data['trans_info'])
                alpha_pred = (alpha_pred.numpy()).squeeze()
                alpha_pred = (alpha_pred * 255).astype('uint8')
                alpha_pred = P.save_alpha_pred(alpha_pred, trimap)
                result.append(alpha_pred)
                if visualization:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img_name = str(time.time()) + '.png'
                    image_save_path = os.path.join(save_path, img_name)
                    cv2.imwrite(image_save_path, alpha_pred)

        return result
    
    @serving
    def serving_method(self, images: list, trimaps:list, **kwargs) -> dict:
        """
        Run as a service.
        """
        images_decode = [P.base64_to_cv2(image) for image in images]

        if trimaps is not None:
            trimap_decoder = [cv2.cvtColor(P.base64_to_cv2(trimap), cv2.COLOR_BGR2GRAY) for trimap in trimaps]
        else:
            trimap_decoder = None

        outputs = self.predict(image_list=images_decode, trimap_list= trimap_decoder, **kwargs)
        
        serving_data = [P.cv2_to_base64(outputs[i]) for i in range(len(outputs))]
        results = {'data': serving_data}

        return results

    @runnable
    def run_cmd(self, argvs: list) -> list:
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
        if args.trimap_path is not None:
            trimap_list = [args.trimap_path]
        else:
            trimap_list = None

        results = self.predict(image_list=[args.input_path], trimap_list=trimap_list, save_path=args.output_dir, visualization=args.visualization)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default="dim_vgg16_matting_output", help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
        self.arg_input_group.add_argument('--trimap_path', type=str, help="path to trimap.")                
                   
    
class Up(nn.Layer):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            input_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias_attr=False)

    def forward(self, x: paddle.Tensor, skip: paddle.Tensor, output_shape: list) -> paddle.Tensor:
        x = F.interpolate(
            x, size=output_shape, mode='bilinear', align_corners=False)
        x = x + skip
        x = self.conv(x)
        x = F.relu(x)

        return x


class Decoder(nn.Layer):
    def __init__(self, input_channels: int, output_channels: list = [64, 128, 256, 512]):
        super().__init__()
        self.deconv6 = nn.Conv2D(
            input_channels, input_channels, kernel_size=1, bias_attr=False)
        self.deconv5 = Up(input_channels, output_channels[-1])
        self.deconv4 = Up(output_channels[-1], output_channels[-2])
        self.deconv3 = Up(output_channels[-2], output_channels[-3])
        self.deconv2 = Up(output_channels[-3], output_channels[-4])
        self.deconv1 = Up(output_channels[-4], 64)

        self.alpha_conv = nn.Conv2D(
            64, 1, kernel_size=5, padding=2, bias_attr=False)

    def forward(self, fea_list: list, shape_list: list) -> paddle.Tensor:
        x = fea_list[-1]
        x = self.deconv6(x)
        x = self.deconv5(x, fea_list[4], shape_list[4])
        x = self.deconv4(x, fea_list[3], shape_list[3])
        x = self.deconv3(x, fea_list[2], shape_list[2])
        x = self.deconv2(x, fea_list[1], shape_list[1])
        x = self.deconv1(x, fea_list[0], shape_list[0])
        alpha = self.alpha_conv(x)
        alpha = F.sigmoid(alpha)

        return alpha


class Refine(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(
            4, 64, kernel_size=3, padding=1, bias_attr=False)
        self.conv2 = layers.ConvBNReLU(
            64, 64, kernel_size=3, padding=1, bias_attr=False)
        self.conv3 = layers.ConvBNReLU(
            64, 64, kernel_size=3, padding=1, bias_attr=False)
        self.alpha_pred = layers.ConvBNReLU(
            64, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        alpha = self.alpha_pred(x)

        return alpha
