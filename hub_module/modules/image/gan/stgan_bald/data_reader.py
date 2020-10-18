#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import print_function
from six.moves import range
from PIL import Image, ImageOps

import gzip
import numpy as np
import argparse
import struct
import os
import paddle
import paddle.fluid as fluid
import random
import sys


def RandomCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = np.random.randint(0, w - crop_w)
    j = np.random.randint(0, h - crop_h)
    return img.crop((i, j, i + crop_w, j + crop_h))


def CentorCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = int((w - crop_w) / 2.0)
    j = int((h - crop_h) / 2.0)
    return img.crop((i, j, i + crop_w, j + crop_h))


def RandomHorizonFlip(img):
    i = np.random.rand()
    if i > 0.5:
        img = ImageOps.mirror(img)
    return img


def get_preprocess_param2(load_size, crop_size):
    x = np.random.randint(0, np.maximum(0, load_size - crop_size))
    y = np.random.randint(0, np.maximum(0, load_size - crop_size))
    flip = np.random.rand() > 0.5
    return {
        "crop_pos": (x, y),
        "flip": flip,
        "load_size": load_size,
        "crop_size": crop_size
    }


def get_preprocess_param4(load_width, load_height, crop_width, crop_height):
    if crop_width == load_width:
        x = 0
        y = 0
    else:
        x = np.random.randint(0, np.maximum(0, load_width - crop_width))
        y = np.random.randint(0, np.maximum(0, load_height - crop_height))
    flip = np.random.rand() > 0.5
    return {"crop_pos": (x, y), "flip": flip}






class celeba_reader_creator():
    ''' read and preprocess dataset'''

    def __init__(self, image, mode="TRAIN"):
        # self.image_dir = image_dir
        self.image = image
        self.mode = mode

        # lines = open(self.image).readlines()
        
        all_num = 1
        train_end = 2 + int(all_num * 0.9)
        test_end = train_end + int(all_num * 0.003)

        all_attr_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
        attr2idx = {}
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i
            
        if self.mode == 'VAL':
            self.batch_size = 1
            self.shuffle = False
            lines = ['' + image + ' -1 -1 1 -1 -1 1 1 1 -1 -1 -1 -1 1']
        else:
            raise NotImplementedError(
                "Wrong Reader MODE: {}, mode must in [TRAIN|TEST|VAL]".format(
                    self.mode))

        self.images = []
        selected_attrs = 'Bald,Bangs,Black_Hair,Blond_Hair,Brown_Hair,Bushy_Eyebrows,Eyeglasses,Male,Mouth_Slightly_Open,Mustache,No_Beard,Pale_Skin,Young'
        attr_names = selected_attrs.split(',')
        
        arr = lines[0].strip().split()
        if "/" in arr[0]:
            name = arr[0][arr[0].rfind('/'):]
            name = name[1:]
        else:
            name = arr[0]
        label = []
        for attr_name in attr_names:
            idx = attr2idx[attr_name]
            label.append(arr[idx + 1] == "1")
        self.images.append((arr[0], label, name))


    def len(self):
        return len(self.images) // self.batch_size

    def make_reader(self, return_name=False):
        # print(self.image_dir, self.list_filename)
        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_3 = []
            batch_out_name = []
            if self.shuffle:
                np.random.shuffle(self.images)
            for file, label, f_name in self.images:
                img = Image.open(file)
                label = np.array(label).astype("float32")
                model_net = "STGAN"
                if model_net == "StarGAN":
                    img = RandomHorizonFlip(img)
                img = CentorCrop(img, 178, 178)
                img = img.resize((128 , 128),
                                 Image.BILINEAR)
                img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
                img = img.transpose([2, 0, 1])

                batch_out_1.append(img)
                batch_out_2.append(label)
                if return_name:
                    batch_out_name.append(int(f_name.split('.')[0]))
                if len(batch_out_1) == self.batch_size:
                    batch_out_3 = np.copy(batch_out_2)
                    if self.shuffle:
                        np.random.shuffle(batch_out_3)
                    if return_name:
                        yield batch_out_1, batch_out_2, batch_out_3, batch_out_name
                        batch_out_name = []
                    else:
                        yield batch_out_1, batch_out_2, batch_out_3
                    batch_out_1 = []
                    batch_out_2 = []
                    batch_out_3 = []

        return reader



