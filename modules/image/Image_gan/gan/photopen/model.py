# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import os

import cv2
import numpy as np
import paddle
from PIL import Image
from PIL import ImageOps
from ppgan.models.generators import SPADEGenerator
from ppgan.utils.filesystem import load
from ppgan.utils.photopen import data_onehot_pro


class PhotoPenPredictor:
    def __init__(self, weight_path, gen_cfg):

        # 初始化模型
        gen = SPADEGenerator(
            gen_cfg.ngf,
            gen_cfg.num_upsampling_layers,
            gen_cfg.crop_size,
            gen_cfg.aspect_ratio,
            gen_cfg.norm_G,
            gen_cfg.semantic_nc,
            gen_cfg.use_vae,
            gen_cfg.nef,
        )
        gen.eval()
        para = load(weight_path)
        if 'net_gen' in para:
            gen.set_state_dict(para['net_gen'])
        else:
            gen.set_state_dict(para)

        self.gen = gen
        self.gen_cfg = gen_cfg

    def run(self, image):
        sem = Image.fromarray(image).convert('L')
        sem = sem.resize((self.gen_cfg.crop_size, self.gen_cfg.crop_size), Image.NEAREST)
        sem = np.array(sem).astype('float32')
        sem = paddle.to_tensor(sem)
        sem = sem.reshape([1, 1, self.gen_cfg.crop_size, self.gen_cfg.crop_size])

        one_hot = data_onehot_pro(sem, self.gen_cfg)
        predicted = self.gen(one_hot)
        pic = predicted.numpy()[0].reshape((3, 256, 256)).transpose((1, 2, 0))
        pic = ((pic + 1.) / 2. * 255).astype('uint8')

        return pic
