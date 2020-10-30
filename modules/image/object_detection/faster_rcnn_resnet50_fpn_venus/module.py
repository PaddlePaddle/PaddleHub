# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import argparse
from collections import OrderedDict
from functools import partial
from math import ceil

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable, serving
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.io.parser import txt_parser
from paddlehub.common.paddle_helper import add_vars_prefix

from faster_rcnn_resnet50_fpn_venus.processor import load_label_info, postprocess, base64_to_cv2
from faster_rcnn_resnet50_fpn_venus.data_feed import test_reader, padding_minibatch
from faster_rcnn_resnet50_fpn_venus.fpn import FPN
from faster_rcnn_resnet50_fpn_venus.resnet import ResNet
from faster_rcnn_resnet50_fpn_venus.rpn_head import AnchorGenerator, RPNTargetAssign, GenerateProposals, FPNRPNHead
from faster_rcnn_resnet50_fpn_venus.bbox_head import MultiClassNMS, BBoxHead, TwoFCHead
from faster_rcnn_resnet50_fpn_venus.bbox_assigner import BBoxAssigner
from faster_rcnn_resnet50_fpn_venus.roi_extractor import FPNRoIAlign


@moduleinfo(
    name="faster_rcnn_resnet50_fpn_venus",
    version="1.0.0",
    type="cv/object_detection",
    summary=
    "Baidu's Faster-RCNN model for object detection, whose backbone is ResNet50, processed with Feature Pyramid Networks",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class FasterRCNNResNet50RPN(hub.Module):
    def _initialize(self):
        # default pretrained model, Faster-RCNN with backbone ResNet50, shape of input tensor is [3, 800, 1333]
        self.default_pretrained_model_path = os.path.join(self.directory, "faster_rcnn_resnet50_fpn_model")

    def context(self, num_classes=708, trainable=True, pretrained=True, phase='train'):
        """
        Distill the Head Features, so as to perform transfer learning.

        Args:
            trainable (bool): whether to set parameters trainable.
            pretrained (bool): whether to load default pretrained model.
            get_prediction (bool): whether to get prediction.
            phase (str): optional choices are 'train' and 'predict'.

        Returns:
             inputs (dict): the input variables.
             outputs (dict): the output variables.
             context_prog (Program): the program to execute transfer learning.
        """
        context_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(context_prog, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name='image', shape=[-1, 3, -1, -1], dtype='float32')
                # backbone
                backbone = ResNet(norm_type='affine_channel', depth=50, feature_maps=[2, 3, 4, 5], freeze_at=2)
                body_feats = backbone(image)
                # fpn
                fpn = FPN(max_level=6, min_level=2, num_chan=256, spatial_scale=[0.03125, 0.0625, 0.125, 0.25])
                var_prefix = '@HUB_{}@'.format(self.name)
                im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32', lod_level=0)
                im_shape = fluid.layers.data(name='im_shape', shape=[3], dtype='float32', lod_level=0)
                body_feat_names = list(body_feats.keys())
                body_feats, spatial_scale = fpn.get_output(body_feats)
                # rpn_head: RPNHead
                rpn_head = self.rpn_head()
                rois = rpn_head.get_proposals(body_feats, im_info, mode=phase)
                # train
                if phase == 'train':
                    gt_bbox = fluid.layers.data(name='gt_bbox', shape=[4], dtype='float32', lod_level=1)
                    is_crowd = fluid.layers.data(name='is_crowd', shape=[1], dtype='int32', lod_level=1)
                    gt_class = fluid.layers.data(name='gt_class', shape=[1], dtype='int32', lod_level=1)
                    rpn_loss = rpn_head.get_loss(im_info, gt_bbox, is_crowd)
                    # bbox_assigner: BBoxAssigner
                    bbox_assigner = self.bbox_assigner(num_classes)
                    outs = fluid.layers.generate_proposal_labels(
                        rpn_rois=rois,
                        gt_classes=gt_class,
                        is_crowd=is_crowd,
                        gt_boxes=gt_bbox,
                        im_info=im_info,
                        batch_size_per_im=bbox_assigner.batch_size_per_im,
                        fg_fraction=bbox_assigner.fg_fraction,
                        fg_thresh=bbox_assigner.fg_thresh,
                        bg_thresh_hi=bbox_assigner.bg_thresh_hi,
                        bg_thresh_lo=bbox_assigner.bg_thresh_lo,
                        bbox_reg_weights=bbox_assigner.bbox_reg_weights,
                        class_nums=bbox_assigner.class_nums,
                        use_random=bbox_assigner.use_random)
                    rois = outs[0]

                roi_extractor = self.roi_extractor()
                roi_feat = roi_extractor(head_inputs=body_feats, rois=rois, spatial_scale=spatial_scale)
                # head_feat
                bbox_head = self.bbox_head(num_classes)
                head_feat = bbox_head.head(roi_feat)
                if isinstance(head_feat, OrderedDict):
                    head_feat = list(head_feat.values())[0]
                if phase == 'train':
                    inputs = {
                        'image': var_prefix + image.name,
                        'im_info': var_prefix + im_info.name,
                        'im_shape': var_prefix + im_shape.name,
                        'gt_class': var_prefix + gt_class.name,
                        'gt_bbox': var_prefix + gt_bbox.name,
                        'is_crowd': var_prefix + is_crowd.name
                    }
                    outputs = {
                        'head_features': var_prefix + head_feat.name,
                        'rpn_cls_loss': var_prefix + rpn_loss['rpn_cls_loss'].name,
                        'rpn_reg_loss': var_prefix + rpn_loss['rpn_reg_loss'].name,
                        'generate_proposal_labels': [var_prefix + var.name for var in outs]
                    }
                elif phase == 'predict':
                    pred = bbox_head.get_prediction(roi_feat, rois, im_info, im_shape)
                    inputs = {
                        'image': var_prefix + image.name,
                        'im_info': var_prefix + im_info.name,
                        'im_shape': var_prefix + im_shape.name
                    }
                    outputs = {
                        'head_features': var_prefix + head_feat.name,
                        'rois': var_prefix + rois.name,
                        'bbox_out': var_prefix + pred.name
                    }
                add_vars_prefix(context_prog, var_prefix)
                add_vars_prefix(startup_program, var_prefix)

                global_vars = context_prog.global_block().vars
                inputs = {key: global_vars[value] for key, value in inputs.items()}
                outputs = {
                    key: global_vars[value] if not isinstance(value, list) else [global_vars[var] for var in value]
                    for key, value in outputs.items()
                }

                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(startup_program)
                if pretrained:

                    def _if_exist(var):
                        if num_classes != 81:
                            if 'bbox_pred' in var.name or 'cls_score' in var.name:
                                return False
                        return os.path.exists(os.path.join(self.default_pretrained_model_path, var.name))

                    fluid.io.load_vars(exe, self.default_pretrained_model_path, predicate=_if_exist)
                return inputs, outputs, context_prog

    def rpn_head(self):
        return FPNRPNHead(
            anchor_generator=AnchorGenerator(
                anchor_sizes=[32, 64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                stride=[16.0, 16.0],
                variance=[1.0, 1.0, 1.0, 1.0]),
            rpn_target_assign=RPNTargetAssign(
                rpn_batch_size_per_im=256,
                rpn_fg_fraction=0.5,
                rpn_negative_overlap=0.3,
                rpn_positive_overlap=0.7,
                rpn_straddle_thresh=0.0),
            train_proposal=GenerateProposals(min_size=0.0, nms_thresh=0.7, post_nms_top_n=2000, pre_nms_top_n=2000),
            test_proposal=GenerateProposals(min_size=0.0, nms_thresh=0.7, post_nms_top_n=1000, pre_nms_top_n=1000),
            anchor_start_size=32,
            num_chan=256,
            min_level=2,
            max_level=6)

    def roi_extractor(self):
        return FPNRoIAlign(
            canconical_level=4, canonical_size=224, max_level=5, min_level=2, box_resolution=7, sampling_ratio=2)

    def bbox_head(self, num_classes):
        return BBoxHead(
            head=TwoFCHead(mlp_dim=1024),
            nms=MultiClassNMS(keep_top_k=100, nms_threshold=0.5, score_threshold=0.05),
            num_classes=num_classes)

    def bbox_assigner(self, num_classes):
        return BBoxAssigner(
            batch_size_per_im=512,
            bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
            bg_thresh_hi=0.5,
            bg_thresh_lo=0.0,
            fg_fraction=0.25,
            fg_thresh=0.5,
            class_nums=num_classes)
