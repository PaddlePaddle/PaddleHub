# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import copy
import os

import cv2
import numpy as np
import paddle
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import resize

import paddlehub as hub
from .model import ESNet_x0_25
from .processor import base64_to_cv2
from .processor import create_operators
from .processor import Topk
from .utils import get_config
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="esnet_x0_25_imagenet",
            type="cv/classification",
            author="paddlepaddle",
            author_email="",
            summary="",
            version="1.0.0")
class Esnet_x0_25_Imagenet:

    def __init__(self):
        self.config = get_config(os.path.join(self.directory, 'ESNet_x0_25.yaml'), show=False)
        self.label_path = os.path.join(self.directory, 'imagenet1k_label_list.txt')
        self.pretrain_path = os.path.join(self.directory, 'ESNet_x0_25_pretrained.pdparams')
        self.config['Infer']['PostProcess']['class_id_map_file'] = self.label_path
        self.model = ESNet_x0_25()
        param_state_dict = paddle.load(self.pretrain_path)
        self.model.set_dict(param_state_dict)
        self.preprocess_funcs = create_operators(self.config["Infer"]["transforms"])

    def classification(self,
                       images: list = None,
                       paths: list = None,
                       batch_size: int = 1,
                       use_gpu: bool = False,
                       top_k: int = 1):
        '''
        Args:
            images (list[numpy.ndarray]): data of images, shape of each is [H, W, C], color space must be BGR.
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            top_k (int): Return top k results.

        Returns:
            res (list[dict]): The classfication results, each result dict contains key 'class_ids', 'scores' and 'label_names'.
        '''
        postprocess_func = Topk(top_k, self.label_path)
        inputs = []
        results = []
        paddle.disable_static()
        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)
        if images == None and paths == None:
            print('No image provided. Please input an image or a image path.')
            return

        if images != None:
            for image in images:
                image = image[:, :, ::-1]
                inputs.append(image)

        if paths != None:
            for path in paths:
                image = cv2.imread(path)[:, :, ::-1]
                inputs.append(image)

        batch_data = []
        for idx, imagedata in enumerate(inputs):
            for process in self.preprocess_funcs:
                imagedata = process(imagedata)
            batch_data.append(imagedata)
            if len(batch_data) >= batch_size or idx == len(inputs) - 1:
                batch_tensor = paddle.to_tensor(batch_data)
                out = self.model(batch_tensor)
                if isinstance(out, list):
                    out = out[0]
                if isinstance(out, dict) and "logits" in out:
                    out = out["logits"]
                if isinstance(out, dict) and "output" in out:
                    out = out["output"]
                result = postprocess_func(out)
                results.extend(result)
                batch_data.clear()
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
        self.args = self.parser.parse_args(argvs)
        results = self.classification(paths=[self.args.input_path],
                                      use_gpu=self.args.use_gpu,
                                      batch_size=self.args.batch_size,
                                      top_k=self.args.top_k)
        return results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.classification(images=images_decode, **kwargs)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.arg_config_group.add_argument('--top_k', type=int, default=1, help='Return top k results.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to input image.")
