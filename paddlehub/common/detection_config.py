#coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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


conf = {
    "ssd": {
        "with_background": True,
        "is_bbox_normalized": True,
        "norm_type": "bn",
    },
    "yolo": {
        "with_background": False,
        "is_bbox_normalized": False,
        "norm_type": "sync_bn",
    },
    "rcnn": {
        "with_background": True,
        "is_bbox_normalized": False,
        "norm_type": "affine_channel",
    }
}

ssd_train_ops = [
    dict(op='DecodeImage', to_rgb=True, with_mixup=False),
    dict(op='NormalizeBox'),
    dict(
        op='RandomDistort',
        brightness_lower=0.875,
        brightness_upper=1.125,
        is_order=True),
    dict(op='ExpandImage', max_ratio=4, prob=0.5),
    dict(
        op='CropImage',
        batch_sampler=[[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                       [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0],
                       [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0],
                       [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0],
                       [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0],
                       [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0],
                       [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]],
        satisfy_all=False,
        avoid_no_bbox=False),
    dict(op='ResizeImage', target_size=300, use_cv2=False, interp=1),
    dict(op='RandomFlipImage', is_normalized=True),
    dict(op='Permute'),
    dict(
        op='NormalizeImage',
        mean=[127.5, 127.5, 127.5],
        std=[127.502231, 127.502231, 127.502231],
        is_scale=False),
    dict(op='ArrangeSSD')
]

ssd_eval_fields = ['image', 'gt_box', 'gt_label', 'im_shape', 'im_id', 'is_difficult']
ssd_eval_ops = [
    dict(op='DecodeImage', to_rgb=True, with_mixup=False),
    dict(op='NormalizeBox'),
    dict(op='ResizeImage', target_size=300, use_cv2=False, interp=1),
    dict(op='Permute'),
    dict(
        op='NormalizeImage',
        mean=[127.5, 127.5, 127.5],
        std=[127.502231, 127.502231, 127.502231],
        is_scale=False),
    dict(op='ArrangeEvalSSD', fields=ssd_eval_fields)
]

ssd_predict_ops = [
    dict(op='DecodeImage', to_rgb=True, with_mixup=False),
    dict(op='ResizeImage', target_size=300, use_cv2=False, interp=1),
    dict(op='Permute'),
    dict(
        op='NormalizeImage',
        mean=[127.5, 127.5, 127.5],
        std=[127.502231, 127.502231, 127.502231],
        is_scale=False),
    dict(op='ArrangeTestSSD')
]

feed_config = {
    "ssd": {
        "train": {
            "fields": ['image', 'gt_box', 'gt_label'],
            "OPS": ssd_train_ops,
            "IS_PADDING": False,
        },
        "dev": {
            "fields": ssd_eval_fields,
            "OPS": ssd_eval_ops,
            "IS_PADDING": False,
        },
        "predict": {
            "fields": ['image', 'im_id', 'im_shape'],
            "OPS": ssd_predict_ops,
            "IS_PADDING": False,
        },
    },
    "yolo": {
        "train": {},
        "dev": {},
        "predict": {},
    },
    "rcnn": {
        "train": {},
        "dev": {},
        "predict": {},
    },
}
