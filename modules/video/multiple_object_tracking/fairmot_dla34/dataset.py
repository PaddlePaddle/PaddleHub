# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import six
from collections.abc import Mapping
from collections import deque

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
from ppdet.data.reader import BaseDataLoader, Compose
from paddle.fluid.dataloader.collate import default_collate_fn
import cv2
from imageio import imread, imwrite
import numpy as np
import paddle

logger = setup_logger(__name__)


@register
@serializable
class MOTVideoStream:
    """
    Load MOT dataset with MOT format from video stream.
    Args:
        video_stream (str): path or url of the video file, default ''.
        keep_ori_im (bool): whether to keep original image, default False.
            Set True when used during MOT model inference while saving
            images or video, or used in DeepSORT.
    """
    def __init__(self, video_stream=None, keep_ori_im=False, **kwargs):
        self.video_stream = video_stream
        self.keep_ori_im = keep_ori_im
        self._curr_iter = 0
        self.transform = None
        try:
            if video_stream == None:
                print('No video stream is specified, please check the --video_stream option.')
                raise FileNotFoundError("No video_stream is specified.")
            self.stream = cv2.VideoCapture(video_stream)
            if not self.stream.isOpened():
                raise Exception("Open video stream Error!")
        except Exception as e:
            print('Failed to read {}.'.format(video_stream))
            raise e

        self.videoframeraw_dir = os.path.splitext(os.path.basename(self.video_stream))[0] + '_raw'
        if not os.path.exists(self.videoframeraw_dir):
            os.makedirs(self.videoframeraw_dir)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self):
        pass

    def __iter__(self):
        ct = 0
        while True:
            ret, frame = self.stream.read()
            if ret:
                imgname = os.path.join(self.videoframeraw_dir, 'frame{}.png'.format(ct))
                cv2.imwrite(imgname, frame)
                image = imread(imgname)
                rec = {'im_id': np.array([ct]), 'im_file': imgname}
                if self.keep_ori_im:
                    rec.update({'keep_ori_im': 1})
                rec['curr_iter'] = self._curr_iter
                self._curr_iter += 1
                ct += 1
                if self.transform:
                    yield self.transform(rec)
                else:
                    yield rec
            else:
                return


@register
@serializable
class MOTImageStream:
    """
    Load MOT dataset with MOT format from image stream.
    Args:
        keep_ori_im (bool): whether to keep original image, default False.
            Set True when used during MOT model inference while saving
            images or video, or used in DeepSORT.
    """
    def __init__(self, sample_num=-1, keep_ori_im=False, **kwargs):
        self.keep_ori_im = keep_ori_im
        self._curr_iter = 0
        self.transform = None
        self.imagequeue = deque()

        self.frameraw_dir = 'inputimages_raw'
        if not os.path.exists(self.frameraw_dir):
            os.makedirs(self.frameraw_dir)

    def add_image(self, image):
        self.imagequeue.append(image)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self):
        pass

    def __iter__(self):
        ct = 0
        while True:
            if self.imagequeue:
                frame = self.imagequeue.popleft()
                imgname = os.path.join(self.frameraw_dir, 'frame{}.png'.format(ct))
                cv2.imwrite(imgname, frame)
                image = imread(imgname)
                rec = {'im_id': np.array([ct]), 'im_file': imgname}
                if self.keep_ori_im:
                    rec.update({'keep_ori_im': 1})
                rec['curr_iter'] = self._curr_iter
                self._curr_iter += 1
                ct += 1
                if self.transform:
                    yield self.transform(rec)
                else:
                    yield rec
            else:
                return


@register
class MOTVideoStreamReader:
    __shared__ = ['num_classes']

    def __init__(self, sample_transforms=[], batch_size=1, drop_last=False, num_classes=1, **kwargs):
        self._sample_transforms = Compose(sample_transforms, num_classes=num_classes)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_classes = num_classes
        self.kwargs = kwargs

    def __call__(
        self,
        dataset,
        worker_num,
    ):
        self.dataset = dataset
        # get data
        self.dataset.set_transform(self._sample_transforms)
        # set kwargs
        self.dataset.set_kwargs(**self.kwargs)

        self.loader = iter(self.dataset)
        return self

    def __len__(self):
        return sys.maxint

    def __iter__(self):
        return self

    def to_tensor(self, batch):
        paddle.disable_static()
        if isinstance(batch, np.ndarray):
            batch = paddle.to_tensor(batch)
        elif isinstance(batch, Mapping):
            batch = {key: self.to_tensor(batch[key]) for key in batch}
        return batch

    def __next__(self):
        try:
            batch = []
            for i in range(self.batch_size):
                batch.append(next(self.loader))
            batch = default_collate_fn(batch)
            return self.to_tensor(batch)

        except StopIteration as e:
            raise e

    def next(self):
        # python2 compatibility
        return self.__next__()
