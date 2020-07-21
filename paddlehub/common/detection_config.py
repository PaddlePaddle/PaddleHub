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
        # "norm_type": "bn",
    },
    "yolo": {
        "with_background": False,
        "is_bbox_normalized": False,
        # "norm_type": "sync_bn",
        "mixup_epoch": 10,
        "num_max_boxes": 50,
    },
    "rcnn": {
        "with_background": True,
        "is_bbox_normalized": False,
        # "norm_type": "affine_channel",
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
    dict(op='ResizeImage', target_size=512, use_cv2=False, interp=1),
    dict(op='RandomFlipImage', is_normalized=True),
    dict(op='Permute'),
    dict(
        op='NormalizeImage',
        mean=[104, 117, 123],
        std=[1, 1, 1],
        is_scale=False),
    dict(op='ArrangeSSD')
]

ssd_eval_fields = [
    'image', 'im_shape', 'im_id', 'gt_box', 'gt_label', 'is_difficult'
]
ssd_eval_ops = [
    dict(op='DecodeImage', to_rgb=True, with_mixup=False),
    dict(op='NormalizeBox'),
    dict(op='ResizeImage', target_size=512, use_cv2=False, interp=1),
    dict(op='Permute'),
    dict(
        op='NormalizeImage',
        mean=[104, 117, 123],
        std=[1, 1, 1],
        is_scale=False),
    dict(op='ArrangeEvalSSD', fields=ssd_eval_fields)
]

ssd_predict_ops = [
    dict(op='DecodeImage', to_rgb=True, with_mixup=False),
    dict(op='ResizeImage', target_size=512, use_cv2=False, interp=1),
    dict(op='Permute'),
    dict(
        op='NormalizeImage',
        mean=[104, 117, 123],
        std=[1, 1, 1],
        is_scale=False),
    dict(op='ArrangeTestSSD')
]

rcnn_train_ops = [
    dict(op='DecodeImage', to_rgb=True),
    dict(op='RandomFlipImage', prob=0.5),
    dict(
        op='NormalizeImage',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
        is_channel_first=False),
    dict(op='ResizeImage', target_size=800, max_size=1333, interp=1),
    dict(op='Permute', to_bgr=False),
    dict(op='ArrangeRCNN'),
]

rcnn_eval_ops = [
    dict(op='DecodeImage', to_rgb=True),
    dict(
        op='NormalizeImage',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
        is_channel_first=False),
    dict(op='ResizeImage', target_size=800, max_size=1333, interp=1),
    dict(op='Permute', to_bgr=False),
    dict(op='ArrangeEvalRCNN'),
]

rcnn_predict_ops = [
    dict(op='DecodeImage', to_rgb=True),
    dict(
        op='NormalizeImage',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
        is_channel_first=False),
    dict(op='ResizeImage', target_size=800, max_size=1333, interp=1),
    dict(op='Permute', to_bgr=False),
    dict(op='ArrangeTestRCNN'),
]

yolo_train_ops = [
    dict(op='DecodeImage', to_rgb=True, with_mixup=True),
    dict(op='MixupImage', alpha=1.5, beta=1.5),
    dict(op='ColorDistort'),
    dict(op='RandomExpand', fill_value=[123.675, 116.28, 103.53]),
    dict(op='RandomCrop'),
    dict(op='RandomFlipImage', is_normalized=False),
    dict(op='Resize', target_dim=608, interp='random'),
    dict(
        op='NormalizePermute',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.120, 57.375]),
    dict(op='NormalizeBox'),
    dict(op='ArrangeYOLO'),
]

yolo_eval_ops = [
    dict(op='DecodeImage', to_rgb=True),
    dict(op='ResizeImage', target_size=608, interp=2),
    dict(
        op='NormalizeImage',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
        is_channel_first=False),
    dict(op='Permute', to_bgr=False),
    dict(op='ArrangeEvalYOLO'),
]

yolo_predict_ops = [
    dict(op='DecodeImage', to_rgb=True),
    dict(op='ResizeImage', target_size=608, interp=2),
    dict(
        op='NormalizeImage',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
        is_channel_first=False),
    dict(op='Permute', to_bgr=False),
    dict(op='ArrangeTestYOLO'),
]

feed_config = {
    "ssd": {
        "train": {
            "fields": ['image', 'gt_box', 'gt_label'],
            "OPS": ssd_train_ops,
            "IS_PADDING": False,
        },
        "dev": {
            # ['image', 'im_shape', 'im_id', 'gt_box', 'gt_label', 'is_difficult']
            "fields": ssd_eval_fields,
            "OPS": ssd_eval_ops,
            "IS_PADDING": False,
        },
        "predict": {
            "fields": ['image', 'im_id', 'im_shape'],
            # "fields": ['image', 'im_id'],
            "OPS": ssd_predict_ops,
            "IS_PADDING": False,
        },
    },
    "rcnn": {
        "train": {
            "fields":
            ['image', 'im_info', 'im_id', 'gt_box', 'gt_label', 'is_crowd'],
            "OPS":
            rcnn_train_ops,
            "IS_PADDING":
            True,
            "COARSEST_STRIDE":
            32,
        },
        "dev": {
            "fields": [
                'image', 'im_info', 'im_id', 'im_shape', 'gt_box', 'gt_label',
                'is_difficult'
            ],
            "OPS":
            rcnn_eval_ops,
            "IS_PADDING":
            True,
            "COARSEST_STRIDE":
            32,
            "USE_PADDED_IM_INFO":
            True,
        },
        "predict": {
            "fields": ['image', 'im_info', 'im_id', 'im_shape'],
            "OPS": rcnn_predict_ops,
            "IS_PADDING": True,
            "COARSEST_STRIDE": 32,
            "USE_PADDED_IM_INFO": True,
        },
    },
    "yolo": {
        "train": {
            "fields": ['image', 'gt_box', 'gt_label', 'gt_score'],
            "OPS": yolo_train_ops,
            "RANDOM_SHAPES": [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        },
        "dev": {
            "fields":
            ['image', 'im_size', 'im_id', 'gt_box', 'gt_label', 'is_difficult'],
            "OPS":
            yolo_eval_ops,
        },
        "predict": {
            "fields": ['image', 'im_size', 'im_id'],
            "OPS": yolo_predict_ops,
        },
    },
}


def get_model_type(module_name):
    if 'yolo' in module_name:
        return 'yolo'
    elif 'ssd' in module_name:
        return 'ssd'
    elif 'rcnn' in module_name:
        return 'rcnn'
    else:
        raise ValueError("module {} not supported".format(module_name))


def get_feed_list(module_name, input_dict, input_dict_pred=None):
    pred_feed_list = None
    if 'yolo' in module_name:
        img = input_dict["image"]
        im_size = input_dict["im_size"]
        feed_list = [img.name, im_size.name]
    elif 'ssd' in module_name:
        image = input_dict["image"]
        # image_shape = input_dict["im_shape"]
        image_shape = input_dict["im_size"]
        feed_list = [image.name, image_shape.name]
    elif 'rcnn' in module_name:
        image = input_dict['image']
        im_info = input_dict['im_info']
        gt_bbox = input_dict['gt_bbox']
        gt_class = input_dict['gt_class']
        is_crowd = input_dict['is_crowd']
        feed_list = [
            image.name, im_info.name, gt_bbox.name, gt_class.name, is_crowd.name
        ]
        assert input_dict_pred is not None
        image = input_dict_pred['image']
        im_info = input_dict_pred['im_info']
        im_shape = input_dict['im_shape']
        pred_feed_list = [image.name, im_info.name, im_shape.name]
    else:
        raise NotImplementedError
    return feed_list, pred_feed_list


def get_mid_feature(module_name, output_dict, output_dict_pred=None):
    feature_pred = None
    if 'yolo' in module_name:
        feature = output_dict['head_features']
    elif 'ssd' in module_name:
        feature = output_dict['body_features']
    elif 'rcnn' in module_name:
        head_feat = output_dict['head_feat']
        rpn_cls_loss = output_dict['rpn_cls_loss']
        rpn_reg_loss = output_dict['rpn_reg_loss']
        generate_proposal_labels = output_dict['generate_proposal_labels']
        feature = [
            head_feat, rpn_cls_loss, rpn_reg_loss, generate_proposal_labels
        ]
        assert output_dict_pred is not None
        head_feat = output_dict_pred['head_feat']
        rois = output_dict_pred['rois']
        feature_pred = [head_feat, rois]
    else:
        raise NotImplementedError
    return feature, feature_pred
