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
import numbers
import os
import sys
from collections import deque
from collections.abc import Mapping

import six
try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
from ppdet.data.reader import BaseDataLoader, Compose
import cv2
from imageio import imread, imwrite
import numpy as np
import paddle
from paddle.framework import core

logger = setup_logger(__name__)


def default_collate_fn(batch):
    """
    Default batch collating function for :code:`paddle.io.DataLoader`,
    get input data as a list of sample datas, each element in list
    if the data of a sample, and sample data should composed of list,
    dictionary, string, number, numpy array and paddle.Tensor, this
    function will parse input data recursively and stack number,
    numpy array and paddle.Tensor datas as batch datas. e.g. for
    following input data:
    [{'image': np.array(shape=[3, 224, 224]), 'label': 1},
     {'image': np.array(shape=[3, 224, 224]), 'label': 3},
     {'image': np.array(shape=[3, 224, 224]), 'label': 4},
     {'image': np.array(shape=[3, 224, 224]), 'label': 5},]


    This default collate function zipped each number and numpy array
    field together and stack each field as the batch field as follows:
    {'image': np.array(shape=[4, 3, 224, 224]), 'label': np.array([1, 3, 4, 5])}
    Args:
        batch(list of sample data): batch should be a list of sample data.

    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, (paddle.Tensor)):
        return paddle.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {key: default_collate_fn([d[key] for d in batch]) for key in sample}
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError("fileds number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))


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
