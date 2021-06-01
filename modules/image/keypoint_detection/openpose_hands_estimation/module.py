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
import base64
import copy
import time
import argparse
from typing import Union
from collections import OrderedDict

import cv2
import paddle
import numpy as np
import paddle.nn as nn
import paddlehub as hub
from skimage.measure import label
from scipy.ndimage.filters import gaussian_filter
from paddlehub.module.module import moduleinfo, runnable, serving
import paddlehub.vision.transforms as T

import openpose_hands_estimation.processor as P


@moduleinfo(
    name="openpose_hands_estimation",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="Openpose_hands_estimation is a hand pose estimation model based on Hand Keypoint Detection in \
            Single Images using Multiview Bootstrapping.",
    version="1.0.0")
class HandPoseModel(nn.Layer):
    """
    HandposeModel

    Args:
        load_checkpoint(str): Checkpoint save path, default is None.
    """

    def __init__(self, load_checkpoint: str = None):
        super(HandPoseModel, self).__init__()

        self.norm_func = T.Normalize(std=[1, 1, 1])
        self.resize_func = P.ResizeScaling()
        self.hand_detect = P.HandDetect()
        self.pad_func = P.PadDownRight()
        self.remove_pad = P.RemovePadding()
        self.draw_pose = P.DrawPose()
        self.draw_hand = P.DrawHandPose()

        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3', \
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']

        block1_0 = OrderedDict([('conv1_1', [3, 64, 3, 1, 1]), ('conv1_2', [64, 64, 3, 1, 1]),
                                ('pool1_stage1', [2, 2, 0]), ('conv2_1', [64, 128, 3, 1, 1]),
                                ('conv2_2', [128, 128, 3, 1, 1]), ('pool2_stage1', [2, 2, 0]),
                                ('conv3_1', [128, 256, 3, 1, 1]), ('conv3_2', [256, 256, 3, 1, 1]),
                                ('conv3_3', [256, 256, 3, 1, 1]), ('conv3_4', [256, 256, 3, 1, 1]),
                                ('pool3_stage1', [2, 2, 0]), ('conv4_1', [256, 512, 3, 1, 1]),
                                ('conv4_2', [512, 512, 3, 1, 1]), ('conv4_3', [512, 512, 3, 1, 1]),
                                ('conv4_4', [512, 512, 3, 1, 1]), ('conv5_1', [512, 512, 3, 1, 1]),
                                ('conv5_2', [512, 512, 3, 1, 1]), ('conv5_3_CPM', [512, 128, 3, 1, 1])])

        block1_1 = OrderedDict([('conv6_1_CPM', [128, 512, 1, 1, 0]), ('conv6_2_CPM', [512, 22, 1, 1, 0])])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                                                 ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                                                 ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                                                 ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                                                 ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                                                 ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                                                 ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])])

        for k in blocks.keys():
            blocks[k] = self.make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'openpose_hand.pdparams')
            self.model_dict = paddle.load(checkpoint)
            self.set_dict(self.model_dict)
            print("load pretrained checkpoint success")
        self.body_model = None

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

    def forward(self, x: paddle.Tensor):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = paddle.concat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = paddle.concat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = paddle.concat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = paddle.concat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = paddle.concat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6

    def hand_estimation(self, handimg: np.ndarray, scale_search: list):
        heatmap_avg = np.zeros((handimg.shape[0], handimg.shape[1], 22))
        for scale in scale_search:
            process = self.resize_func(handimg, scale)
            imageToTest_padded, pad = self.pad_func(process)
            process = self.norm_func(imageToTest_padded)
            process = np.ascontiguousarray(np.transpose(process[:, :, :, np.newaxis], (3, 2, 0, 1))).astype("float32")
            data = self.forward(paddle.to_tensor(process))
            data = data.numpy()
            heatmap = self.remove_pad(data, imageToTest_padded, handimg, pad)
            heatmap_avg += heatmap / len(scale_search)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > 0.05, dtype=np.uint8)
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = P.npmax(map_ori)
            all_peaks.append([x, y])

        return np.array(all_peaks)

    def predict(self,
                img: Union[str, np.ndarray],
                save_path: str = 'openpose_hand',
                scale: list = [0.5, 1.0, 1.5, 2.0],
                visualization: bool = True):
        self.eval()
        self.visualization = visualization
        if isinstance(img, str):
            org_img = cv2.imread(img)
        else:
            org_img = img

        if not self.body_model:
            self.body_model = hub.Module(name='openpose_body_estimation')
            self.body_model.eval()

        body_result = self.body_model.predict(org_img)
        hands_list = self.hand_detect(body_result['candidate'], body_result['subset'], org_img)

        all_hand_peaks = []

        for x, y, w, is_left in hands_list:
            peaks = self.hand_estimation(org_img[y:y + w, x:x + w, :], scale)
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)
        canvas = copy.deepcopy(org_img)
        canvas = self.draw_pose(
            canvas,
            body_result['candidate'],
            body_result['subset'],
        )
        canvas = self.draw_hand(canvas, all_hand_peaks)
        if self.visualization:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img_name = str(time.time()) + '.png'
            save_path = os.path.join(save_path, img_name)
            cv2.imwrite(save_path, canvas)

        results = {'all_hand_peaks': all_hand_peaks, 'data': canvas}

        return results

    @serving
    def serving_method(self, images: list, **kwargs):
        """
        Run as a service.
        """
        images_decode = [P.base64_to_cv2(image) for image in images]
        results = self.predict(img=images_decode[0], **kwargs)
        final = {}
        final['all_hand_peaks'] = [peak.tolist() for peak in results['all_hand_peaks']]
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
        results = self.predict(
            img=args.input_path, save_path=args.output_dir, scale=args.scale, visualization=args.visualization)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='openpose_hand', help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--scale', type=list, default=[0.5, 1.0, 1.5, 2.0], help="The search scale for openpose hands model.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
