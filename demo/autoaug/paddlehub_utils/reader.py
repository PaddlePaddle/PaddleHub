# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""

Authors: lvhaijun01@baidu.com
Date:     2019-06-30 00:10
"""
import re
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import six
import cv2
import os
import paddle
import paddlehub.vision.transforms as transforms
from PIL import ImageFile
from auto_augment.autoaug.transform.autoaug_transform import AutoAugTransform
ImageFile.LOAD_TRUNCATED_IMAGES = True
__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class PbaAugment(object):
    """
    pytorch 分类 PbaAugment transform
    """

    def __init__(self,
                 input_size: int = 224,
                 scale_size: int = 256,
                 normalize: Optional[list] = None,
                 pre_transform: bool = True,
                 stage: str = "search",
                 **kwargs) -> None:
        """

        Args:
            input_size:
            scale_size:
            normalize:
            pre_transform:
            **kwargs:
        """

        if normalize is None:
            normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

        policy = kwargs["policy"]
        assert stage in ["search", "train"]
        train_epochs = kwargs["hp_policy_epochs"]
        self.auto_aug_transform = AutoAugTransform.create(policy, stage=stage, train_epochs=train_epochs)
        #self.auto_aug_transform = PbtAutoAugmentClassiferTransform(conf)
        if pre_transform:
            self.pre_transform = transforms.Resize(input_size)

        self.post_transform = transforms.Compose(
            transforms=[transforms.Permute(),
                        transforms.Normalize(**normalize, channel_first=True)],
            channel_first=False)
        self.cur_epoch = 0

    def set_epoch(self, indx: int) -> None:
        """

        Args:
            indx:

        Returns:

        """
        self.auto_aug_transform.set_epoch(indx)

    def reset_policy(self, new_hparams: dict) -> None:
        """

        Args:
            new_hparams:

        Returns:

        """
        self.auto_aug_transform.reset_policy(new_hparams)

    def __call__(self, img: np.ndarray):
        """

        Args:
            img: PIL image
        Returns:

        """
        # tensform resize
        if self.pre_transform:
            img = self.pre_transform(img)

        img = self.auto_aug_transform.apply(img)
        img = img.astype(np.uint8)
        img = self.post_transform(img)
        return img


class PicRecord(object):
    """
    PicRecord
    """

    def __init__(self, row: list) -> None:
        """

        Args:
            row:
        """
        self._data = row

    @property
    def sub_path(self) -> str:
        """

        Returns:

        """
        return self._data[0]

    @property
    def label(self) -> str:
        """

        Returns:

        """
        return self._data[1]


class PicReader(paddle.io.Dataset):
    """
    PicReader
    """

    def __init__(self,
                 root_path: str,
                 list_file: str,
                 meta: bool = False,
                 transform: Optional[callable] = None,
                 class_to_id_dict: Optional[dict] = None,
                 cache_img: bool = False,
                 **kwargs) -> None:
        """

        Args:
            root_path:
            list_file:
            meta:
            transform:
            class_to_id_dict:
            cache_img:
            **kwargs:
        """

        self.root_path = root_path
        self.list_file = list_file
        self.transform = transform
        self.meta = meta
        self.class_to_id_dict = class_to_id_dict
        self.train_type = kwargs["conf"].get("train_type", "single_label")
        self.class_num = kwargs["conf"].get("class_num", 0)

        self._parse_list(**kwargs)
        self.cache_img = cache_img
        self.cache_img_buff = dict()
        if self.cache_img:
            self._get_all_img(**kwargs)

    def _get_all_img(self, **kwargs) -> None:
        """
        缓存图片进行预resize, 减少内存占用

        Returns:

        """

        scale_size = kwargs.get("scale_size", 256)

        for idx in range(len(self)):
            record = self.pic_list[idx]
            relative_path = record.sub_path
            if self.root_path is not None:
                image_path = os.path.join(self.root_path, relative_path)
            else:
                image_path = relative_path
            try:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (scale_size, scale_size))
                self.cache_img_buff[image_path] = img
            except BaseException:
                print("img_path:{} can not by cv2".format(image_path).format(image_path))

                pass

    def _load_image(self, directory: str) -> np.ndarray:
        """

        Args:
            directory:

        Returns:

        """

        if not self.cache_img:
            img = cv2.imread(directory, cv2.IMREAD_COLOR).astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.open(directory).convert('RGB')
        else:
            if directory in self.cache_img_buff:
                img = self.cache_img_buff[directory]
            else:
                img = cv2.imread(directory, cv2.IMREAD_COLOR).astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # img = Image.open(directory).convert('RGB')
        return img

    def _parse_list(self, **kwargs) -> None:
        """

        Args:
            **kwargs:

        Returns:

        """
        delimiter = kwargs.get("delimiter", " ")
        self.pic_list = []

        with open(self.list_file) as f:
            lines = f.read().splitlines()
            print("PicReader:: found {} picture in `{}'".format(len(lines), self.list_file))
            for i, line in enumerate(lines):
                record = re.split(delimiter, line)
                # record = line.split()
                assert len(record) == 2, "length of record is not 2!"

                if not os.path.splitext(record[0])[1]:
                    # 适配线上分类数据转无后缀的情况
                    record[0] = record[0] + ".jpg"

                # 线上单标签情况兼容多标签，后续需去除
                record[1] = re.split(",", record[1])[0]

                self.pic_list.append(PicRecord(record))

    def __getitem__(self, index: int):
        """

        Args:
            index:

        Returns:

        """
        record = self.pic_list[index]

        return self.get(record)

    def get(self, record: PicRecord) -> tuple:
        """

        Args:
            record:

        Returns:

        """
        relative_path = record.sub_path
        if self.root_path is not None:
            image_path = os.path.join(self.root_path, relative_path)
        else:
            image_path = relative_path

        img = self._load_image(image_path)
        # print("org img sum:{}".format(np.sum(np.asarray(img))))

        process_data = self.transform(img)

        if self.train_type == "single_label":
            if self.class_to_id_dict:
                label = self.class_to_id_dict[record.label]
            else:
                label = int(record.label)
        elif self.train_type == "multi_labels":
            label_tensor = np.zeros((1, self.class_num))
            for label in record.label.split(","):
                label_tensor[0, int(label)] = 1
            label_tensor = np.squeeze(label_tensor)
            label = label_tensor

        if self.meta:
            return process_data, label, relative_path
        else:
            return process_data, label

    def __len__(self) -> int:
        """

        Returns:

        """
        return len(self.pic_list)

    def set_meta(self, meta: bool) -> None:
        """

        Args:
            meta:

        Returns:

        """
        self.meta = meta

    def set_epoch(self, epoch: int) -> None:
        """

        Args:
            epoch:

        Returns:

        """
        if self.transform is not None:
            self.transform.set_epoch(epoch)

    # only use in search
    def reset_policy(self, new_hparams: dict) -> None:
        """

        Args:
            new_hparams:

        Returns:

        """
        if self.transform is not None:
            self.transform.reset_policy(new_hparams)


def _parse(value: str, function: callable, fmt: str) -> None:
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        six.raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_file: str) -> dict:
    """ Parse the classes file.
    """
    result = {}
    with open(csv_file) as csv_reader:
        for line, row in enumerate(csv_reader):
            try:
                class_name = row.strip()
                # print(class_id, class_name)
            except ValueError:
                six.raise_from(ValueError('line {}: format should be \'class_name\''.format(line)), None)

            class_id = _parse(line, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
    return result


def _init_loader(hparams: dict, TrainTransform=None) -> tuple:
    """

    Args:
        hparams:

    Returns:

    """
    train_data_root = hparams.data_config.train_img_prefix
    val_data_root = hparams.data_config.val_img_prefix
    train_list = hparams.data_config.train_ann_file
    val_list = hparams.data_config.val_ann_file
    input_size = hparams.task_config.classifier.input_size
    scale_size = hparams.task_config.classifier.scale_size
    search_space = hparams.search_space
    search_space["task_type"] = hparams.task_config.task_type
    epochs = hparams.task_config.classifier.epochs
    no_cache_img = hparams.task_config.classifier.get("no_cache_img", False)

    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    if TrainTransform is None:
        TrainTransform = PbaAugment(
            input_size=input_size,
            scale_size=scale_size,
            normalize=normalize,
            policy=search_space,
            hp_policy_epochs=epochs,
        )
    delimiter = hparams.data_config.delimiter
    kwargs = dict(conf=hparams, delimiter=delimiter)

    if hparams.task_config.classifier.use_class_map:
        class_to_id_dict = _read_classes(label_list=hparams.data_config.label_list)
    else:
        class_to_id_dict = None
    train_data = PicReader(
        root_path=train_data_root,
        list_file=train_list,
        transform=TrainTransform,
        class_to_id_dict=class_to_id_dict,
        cache_img=not no_cache_img,
        **kwargs)

    val_data = PicReader(
        root_path=val_data_root,
        list_file=val_list,
        transform=transforms.Compose(
            transforms=[
                transforms.Resize((224, 224)),
                transforms.Permute(),
                transforms.Normalize(**normalize, channel_first=True)
            ],
            channel_first=False),
        class_to_id_dict=class_to_id_dict,
        cache_img=not no_cache_img,
        **kwargs)

    return train_data, val_data
