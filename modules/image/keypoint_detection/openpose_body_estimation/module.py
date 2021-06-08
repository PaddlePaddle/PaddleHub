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

import os
import time
import copy
import base64
import argparse
from typing import Union
from collections import OrderedDict

import cv2
import paddle
import paddle.nn as nn
import numpy as np
from paddlehub.module.module import moduleinfo, runnable, serving
import paddlehub.vision.transforms as T
import openpose_body_estimation.processor as P


@moduleinfo(
    name="openpose_body_estimation",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="Openpose_body_estimation is a body pose estimation model based on Realtime Multi-Person 2D Pose \
            Estimation using Part Affinity Fields.",
    version="1.0.0")
class BodyPoseModel(nn.Layer):
    """
    BodyposeModel

    Args:
        load_checkpoint(str): Checkpoint save path, default is None.
    """

    def __init__(self, load_checkpoint: str = None):
        super(BodyPoseModel, self).__init__()

        self.resize_func = P.ResizeScaling()
        self.norm_func = T.Normalize(std=[1, 1, 1])
        self.pad_func = P.PadDownRight()
        self.remove_pad = P.RemovePadding()
        self.get_peak = P.GetPeak()
        self.get_connection = P.Connection()
        self.get_candidate = P.Candidate()
        self.draw_pose = P.DrawPose()

        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1', \
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2', \
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1', \
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([('conv1_1', [3, 64, 3, 1, 1]), ('conv1_2', [64, 64, 3, 1, 1]), ('pool1_stage1', [2, 2,
                                                                                                               0]),
                              ('conv2_1', [64, 128, 3, 1, 1]), ('conv2_2', [128, 128, 3, 1, 1]),
                              ('pool2_stage1', [2, 2, 0]), ('conv3_1', [128, 256, 3, 1, 1]),
                              ('conv3_2', [256, 256, 3, 1, 1]), ('conv3_3', [256, 256, 3, 1, 1]),
                              ('conv3_4', [256, 256, 3, 1, 1]), ('pool3_stage1', [2, 2, 0]),
                              ('conv4_1', [256, 512, 3, 1, 1]), ('conv4_2', [512, 512, 3, 1, 1]),
                              ('conv4_3_CPM', [512, 256, 3, 1, 1]), ('conv4_4_CPM', [256, 128, 3, 1, 1])])

        block1_1 = OrderedDict([('conv5_1_CPM_L1', [128, 128, 3, 1, 1]), ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                                ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]), ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                                ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])])

        block1_2 = OrderedDict([('conv5_1_CPM_L2', [128, 128, 3, 1, 1]), ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                                ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]), ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                                ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = self.make_layers(block0, no_relu_layers)

        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                                                   ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                                                   ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])])

            blocks['block%d_2' % i] = OrderedDict([('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                                                   ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                                                   ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                                                   ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])])

        for k in blocks.keys():
            blocks[k] = self.make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'openpose_body.pdparams')
            self.model_dict = paddle.load(checkpoint)
            self.set_dict(self.model_dict)
            print("load pretrained checkpoint success")

    def make_layers(self, block: dict, no_relu_layers: list):
        layers = []
        for layer_name, v in block.items():
            if 'pool' in layer_name:
                layer = nn.MaxPool2D(kernel_size=v[0], stride=v[1], padding=v[2])
                layers.append((layer_name, layer))
            else:
                conv2d = nn.Conv2D(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                layers.append((layer_name, conv2d))
                if layer_name not in no_relu_layers:
                    layers.append(('relu_' + layer_name, nn.ReLU()))
        layers = tuple(layers)
        return nn.Sequential(*layers)

    def transform(self, orgimg: np.ndarray, scale_search: float = 0.5):
        process = self.resize_func(orgimg, scale_search)
        imageToTest_padded, pad = self.pad_func(process)
        process = self.norm_func(imageToTest_padded)
        process = np.ascontiguousarray(np.transpose(process[:, :, :, np.newaxis], (3, 2, 0, 1))).astype("float32")

        return process, imageToTest_padded, pad

    def forward(self, x: paddle.Tensor):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = paddle.concat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = paddle.concat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = paddle.concat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = paddle.concat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = paddle.concat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2

    def predict(self, img: Union[str, np.ndarray], save_path: str = "openpose_body", visualization: bool = True):
        self.eval()
        self.visualization = visualization
        if isinstance(img, str):
            orgImg = cv2.imread(img)
        else:
            orgImg = img
        data, imageToTest_padded, pad = self.transform(orgImg)
        Mconv7_stage6_L1, Mconv7_stage6_L2 = self.forward(paddle.to_tensor(data))
        Mconv7_stage6_L1 = Mconv7_stage6_L1.numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.numpy()

        heatmap_avg = self.remove_pad(Mconv7_stage6_L2, imageToTest_padded, orgImg, pad)
        paf_avg = self.remove_pad(Mconv7_stage6_L1, imageToTest_padded, orgImg, pad)

        all_peaks = self.get_peak(heatmap_avg)
        connection_all, special_k = self.get_connection(all_peaks, paf_avg, orgImg)
        candidate, subset = self.get_candidate(all_peaks, connection_all, special_k)

        canvas = copy.deepcopy(orgImg)
        canvas = self.draw_pose(canvas, candidate, subset)
        if self.visualization:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img_name = str(time.time()) + '.png'
            save_path = os.path.join(save_path, img_name)
            cv2.imwrite(save_path, canvas)

        results = {'candidate': candidate, 'subset': subset, 'data': canvas}

        return results

    @serving
    def serving_method(self, images: list, **kwargs):
        """
        Run as a service.
        """
        images_decode = [P.base64_to_cv2(image) for image in images]
        results = self.predict(img=images_decode[0], **kwargs)
        final = {}
        final['candidate'] = P.cv2_to_base64(results['candidate'])
        final['subset'] = P.cv2_to_base64(results['subset'])
        final['data'] = P.cv2_to_base64(results['data'])

        return final

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
        results = self.predict(img=args.input_path, save_path=args.output_dir, visualization=args.visualization)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='openpose_body', help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
