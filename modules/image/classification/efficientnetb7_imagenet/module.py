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
from __future__ import absolute_import
from __future__ import division

import argparse
import ast
import os

import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

from .data_feed import reader
from .processor import base64_to_cv2
from .processor import postprocess
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="efficientnetb7_imagenet",
            type="CV/image_classification",
            author="paddlepaddle",
            author_email="paddle-dev@baidu.com",
            summary="EfficientNetB7 is a image classfication model, this module is trained with imagenet datasets.",
            version="1.2.0")
class EfficientNetB7ImageNet:

    def __init__(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "efficientnetb7_imagenet_infer_model",
                                                          "model")
        label_file = os.path.join(self.directory, "label_list.txt")
        with open(label_file, 'r', encoding='utf-8') as file:
            self.label_list = file.read().split("\n")[:-1]
        self._set_config()

    def get_expected_image_width(self):
        return 224

    def get_expected_image_height(self):
        return 224

    def get_pretrained_images_mean(self):
        im_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3)
        return im_mean

    def get_pretrained_images_std(self):
        im_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3)
        return im_std

    def _set_config(self):
        """
        predictor config setting
        """
        model = self.default_pretrained_model_path + '.pdmodel'
        params = self.default_pretrained_model_path + '.pdiparams'
        cpu_config = Config(model, params)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = Config(model, params)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_predictor(gpu_config)

    def classification(self, images=None, paths=None, batch_size=1, use_gpu=False, top_k=1):
        """
        API for image classification.

        Args:
            images (list[numpy.ndarray]): data of images, shape of each is [H, W, C], color space must be BGR.
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            top_k (int): Return top k results.

        Returns:
            res (list[dict]): The classfication results.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = list()
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['image'] for data in batch_data])

            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(batch_image.copy())
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])

            out = postprocess(data_out=output_handle.copy_to_cpu(), label_list=self.label_list, top_k=top_k)
            res += out
        return res

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.classify(images=images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
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
        results = self.classify(paths=[args.input_path], batch_size=args.batch_size, use_gpu=args.use_gpu)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU or not.")
        self.arg_config_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")
        self.arg_config_group.add_argument('--top_k', type=ast.literal_eval, default=1, help="Return top k results.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")


if __name__ == '__main__':
    b7 = EfficientNetB7ImageNet()
    b7.context()
    import cv2
    test_image = [cv2.imread('dog.jpeg')]
    res = b7.classification(images=test_image)
    print(res)
    res = b7.classification(paths=['dog.jpeg'])
    print(res)
    res = b7.classification(images=test_image)
    print(res)
    res = b7.classify(images=test_image)
    print(res)
