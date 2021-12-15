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

from modnet_resnet50vd_matting.resnet import ResNet50_vd
import modnet_resnet50vd_matting.processor as P


@moduleinfo(
    name="modnet_resnet50vd_matting", 
    type="CV/matting", 
    author="paddlepaddle",
    summary="modnet_resnet50vd_matting is a matting model",  
    version="1.0.0"  
)
class MODNetResNet50Vd(nn.Layer):
    """
    The MODNet implementation based on PaddlePaddle.

    The original article refers to
    Zhanghan Ke, et, al. "Is a Green Screen Really Necessary for Real-Time Portrait Matting?"
    (https://arxiv.org/pdf/2011.11961.pdf).

    Args:
        hr_channels(int, optional): The channels of high resolutions branch. Defautl: None.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.
    """

    def __init__(self, hr_channels:int = 32, pretrained=None):
        super(MODNetResNet50Vd, self).__init__()

        self.backbone = ResNet50_vd()
        self.pretrained = pretrained

        self.head = MODNetHead(
            hr_channels=hr_channels, backbone_channels=self.backbone.feat_channels)
        self.blurer = GaussianBlurLayer(1, 3)
        self.transforms = P.Compose([P.LoadImages(), P.ResizeByShort(), P.ResizeToIntMult(), P.Normalize()])

        if pretrained is not None:
            model_dict = paddle.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        else:
            checkpoint = os.path.join(self.directory, 'modnet-resnet50_vd.pdparams')
            model_dict = paddle.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")
    
    def preprocess(self, img: Union[str, np.ndarray] , transforms: Callable, trimap: Union[str, np.ndarray] = None):
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
    
    def forward(self, inputs: dict):
        x = inputs['img']
        feat_list = self.backbone(x)
        y = self.head(inputs=inputs, feat_list=feat_list)
        return y
    
    def predict(self, image_list: list, trimap_list: list = None, visualization: bool =False, save_path: str = "modnet_resnet50vd_matting_output"):
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
    def serving_method(self, images: list, trimaps:list = None, **kwargs):
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
            '--output_dir', type=str, default="modnet_resnet50vd_matting_output", help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
        self.arg_input_group.add_argument('--trimap_path', type=str, default=None, help="path to trimap.")
                
                   
    
class MODNetHead(nn.Layer):
    """
    Segmentation head.
    """
    def __init__(self, hr_channels: int, backbone_channels: int):
        super().__init__()

        self.lr_branch = LRBranch(backbone_channels)
        self.hr_branch = HRBranch(hr_channels, backbone_channels)
        self.f_branch = FusionBranch(hr_channels, backbone_channels)

    def forward(self, inputs: paddle.Tensor, feat_list: list) -> paddle.Tensor:
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(feat_list)
        pred_detail, hr2x = self.hr_branch(inputs['img'], enc2x, enc4x, lr8x)
        pred_matte = self.f_branch(inputs['img'], lr8x, hr2x)
        return pred_matte



class FusionBranch(nn.Layer):
    def __init__(self, hr_channels: int, enc_channels: int):
        super().__init__()
        self.conv_lr4x = Conv2dIBNormRelu(
            enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = Conv2dIBNormRelu(
            2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(
                hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                int(hr_channels / 2),
                1,
                1,
                stride=1,
                padding=0,
                with_ibn=False,
                with_relu=False))

    def forward(self, img: paddle.Tensor, lr8x: paddle.Tensor, hr2x: paddle.Tensor) -> paddle.Tensor:
        lr4x = F.interpolate(
            lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(
            lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(paddle.concat((lr2x, hr2x), axis=1))
        f = F.interpolate(
            f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(paddle.concat((f, img), axis=1))
        pred_matte = F.sigmoid(f)

        return pred_matte


class HRBranch(nn.Layer):
    """
    High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels: int, enc_channels:int):
        super().__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(
            enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(
            hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(
            enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(
            2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(
                2 * hr_channels + enc_channels[2] + 3,
                2 * hr_channels,
                3,
                stride=1,
                padding=1),
            Conv2dIBNormRelu(
                2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * hr_channels, hr_channels, 3, stride=1, padding=1))

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(
                2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1))

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(
                hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                hr_channels,
                1,
                1,
                stride=1,
                padding=0,
                with_ibn=False,
                with_relu=False))

    def forward(self, img: paddle.Tensor, enc2x: paddle.Tensor, enc4x: paddle.Tensor, lr8x: paddle.Tensor) -> paddle.Tensor:
        img2x = F.interpolate(
            img, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(
            img, scale_factor=1 / 4, mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(paddle.concat((img2x, enc2x), axis=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(paddle.concat((hr4x, enc4x), axis=1))

        lr4x = F.interpolate(
            lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(paddle.concat((hr4x, lr4x, img4x), axis=1))

        hr2x = F.interpolate(
            hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(paddle.concat((hr2x, enc2x), axis=1))
        pred_detail = None
        return pred_detail, hr2x


class LRBranch(nn.Layer):
    """
    Low Resolution Branch of MODNet
    """
    def __init__(self, backbone_channels: int):
        super().__init__()
        self.se_block = SEBlock(backbone_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(
            backbone_channels[4], backbone_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(
            backbone_channels[3], backbone_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(
            backbone_channels[2],
            1,
            3,
            stride=2,
            padding=1,
            with_ibn=False,
            with_relu=False)

    def forward(self, feat_list: list) -> List[paddle.Tensor]:
        enc2x, enc4x, enc32x = feat_list[0], feat_list[1], feat_list[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(
            enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(
            lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if self.training:
            lr = self.conv_lr(lr8x)
            pred_semantic = F.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x]


class IBNorm(nn.Layer):
    """
    Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.bnorm_channels = in_channels // 2
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2D(self.bnorm_channels)
        self.inorm = nn.InstanceNorm2D(self.inorm_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        bn_x = self.bnorm(x[:, :self.bnorm_channels, :, :])
        in_x = self.inorm(x[:, self.bnorm_channels:, :, :])

        return paddle.concat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Layer):
    """
    Convolution + IBNorm + Relu
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation:int = 1,
                 groups: int = 1,
                 bias_attr: paddle.ParamAttr = None,
                 with_ibn: bool = True,
                 with_relu: bool = True):

        super().__init__()

        layers = [
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias_attr)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))

        if with_relu:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.layers(x)


class SEBlock(nn.Layer):
    """
    SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, num_channels: int, reduction:int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Sequential(
            nn.Conv2D(
                num_channels,
                int(num_channels // reduction),
                1,
                bias_attr=False), nn.ReLU(),
            nn.Conv2D(
                int(num_channels // reduction),
                num_channels,
                1,
                bias_attr=False), nn.Sigmoid())

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        w = self.pool(x)
        w = self.conv(w)
        return w * x


class GaussianBlurLayer(nn.Layer):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels: int, kernel_size: int):
        """
        Args:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.Pad2D(int(self.kernel_size / 2), mode='reflect'),
            nn.Conv2D(
                channels,
                channels,
                self.kernel_size,
                stride=1,
                padding=0,
                bias_attr=False,
                groups=channels))

        self._init_kernel()
        self.op[1].weight.stop_gradient = True

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x (paddle.Tensor): input 4D tensor
        Returns:
            paddle.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(
                      self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = int(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)
        kernel = kernel.astype('float32')
        kernel = kernel[np.newaxis, np.newaxis, :, :]
        paddle.assign(kernel, self.op[1].weight)