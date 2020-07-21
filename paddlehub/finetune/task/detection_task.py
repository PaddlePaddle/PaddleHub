#coding:utf-8
#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import OrderedDict
import numpy as np
import six
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA

from .base_task import BaseTask
from ...contrib.ppdet.utils.eval_utils import eval_results
from ...common import detection_config as dconf
from paddlehub.common.paddle_helper import clone_program

feed_var_def = [
    {
        'name': 'im_info',
        'shape': [3],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_id',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 0
    },
    {
        'name': 'gt_box',
        'shape': [4],
        'dtype': 'float32',
        'lod_level': 1
    },
    {
        'name': 'gt_label',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'is_crowd',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'gt_mask',
        'shape': [2],
        'dtype': 'float32',
        'lod_level': 3
    },
    {
        'name': 'is_difficult',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'gt_score',
        'shape': [1],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_shape',
        'shape': [3],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_size',
        'shape': [2],
        'dtype': 'int32',
        'lod_level': 0
    },
]


class Feed(object):
    def __init__(self):
        self.dataset = None
        self.with_background = True


class DetectionTask(BaseTask):
    def __init__(self,
                 data_reader,
                 num_classes,
                 feed_list,
                 feature,
                 model_type='ssd',
                 predict_feature=None,
                 predict_feed_list=None,
                 startup_program=None,
                 config=None,
                 hidden_units=None,
                 metrics_choices="default"):
        if metrics_choices == "default":
            metrics_choices = ["ap"]

        main_program = feature[0].block.program
        super(DetectionTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        if predict_feature is not None:
            main_program = predict_feature[0].block.program
            self._predict_base_main_program = clone_program(
                main_program, for_test=False)
        else:
            self._predict_base_main_program = None
        self._predict_base_feed_list = predict_feed_list
        self.feature = feature
        self.predict_feature = predict_feature
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.model_type = model_type

    @property
    def base_main_program(self):
        if not self.is_train_phase and self._predict_base_main_program is not None:
            return self._predict_base_main_program
        return self._base_main_program

    @property
    def base_feed_list(self):
        if not self.is_train_phase and self._predict_base_feed_list is not None:
            return self._predict_base_feed_list
        return self._base_feed_list

    @property
    def base_feed_var_list(self):
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in self.base_feed_list]

    @property
    def return_numpy(self):
        return 2  # return lod tensor

    def _add_label_by_fields(self, idx_list):
        feed_var_map = {var['name']: var for var in feed_var_def}
        # tensor padding with 0 is used instead of LoD tensor when
        # num_max_boxes is set
        num_max_boxes = dconf.conf[self.model_type].get('num_max_boxes', None)
        if num_max_boxes is not None:
            feed_var_map['gt_label']['shape'] = [num_max_boxes]
            feed_var_map['gt_score']['shape'] = [num_max_boxes]
            feed_var_map['gt_box']['shape'] = [num_max_boxes, 4]
            feed_var_map['is_difficult']['shape'] = [num_max_boxes]
            feed_var_map['gt_label']['lod_level'] = 0
            feed_var_map['gt_score']['lod_level'] = 0
            feed_var_map['gt_box']['lod_level'] = 0
            feed_var_map['is_difficult']['lod_level'] = 0

        if self.is_train_phase:
            fields = dconf.feed_config[self.model_type]['train']['fields']
        elif self.is_test_phase:
            fields = dconf.feed_config[self.model_type]['dev']['fields']
        else:  # Cannot go to here
            # raise RuntimeError("Cannot go to _add_label in predict phase")
            fields = dconf.feed_config[self.model_type]['predict']['fields']

        labels = []
        for i in idx_list:
            key = fields[i]
            l = fluid.layers.data(
                name=feed_var_map[key]['name'],
                shape=feed_var_map[key]['shape'],
                dtype=feed_var_map[key]['dtype'],
                lod_level=feed_var_map[key]['lod_level'])
            labels.append(l)
        return labels

    def _ssd_build_net(self):
        feature_list = self.feature
        image = self.base_feed_var_list[0]

        # fix input size according to its module
        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=feature_list,
            image=image,
            num_classes=self.num_classes,
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2.],
                           [2.]],
            base_size=512,  # 300,
            min_sizes=[20.0, 51.0, 133.0, 215.0, 296.0, 378.0,
                       460.0],  # [60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[51.0, 133.0, 215.0, 296.0, 378.0, 460.0,
                       542.0],  # [[], 150.0, 195.0, 240.0, 285.0, 300.0],
            steps=[8, 16, 32, 64, 128, 256, 512],
            min_ratio=15,
            max_ratio=90,
            kernel_size=3,
            offset=0.5,
            flip=True,
            pad=1,
        )

        self.env.mid_vars = [mbox_locs, mbox_confs, box, box_var]

        nmsed_out = fluid.layers.detection_output(
            mbox_locs,
            mbox_confs,
            box,
            box_var,
            background_label=0,
            nms_threshold=0.45,
            nms_top_k=400,
            keep_top_k=200,
            score_threshold=0.01,
            nms_eta=1.0)

        if self.is_predict_phase:  # add im_id
            self.env.labels = self._ssd_add_label()
        return [nmsed_out]

    def _ssd_add_label(self):
        # train: 'gt_box', 'gt_label'
        # dev: 'im_shape', 'im_id', 'gt_box', 'gt_label', 'is_difficult'
        if self.is_train_phase:
            idx_list = [1, 2]  # 'gt_box', 'gt_label'
        elif self.is_test_phase:
            # xTodo: remove 'im_shape' when using new module
            idx_list = [2, 3, 4,
                        5]  # 'im_id', 'gt_box', 'gt_label', 'is_difficult'
        else:
            idx_list = [1]  # im_id
        return self._add_label_by_fields(idx_list)

    def _ssd_add_loss(self):
        if self.is_train_phase:
            gt_box = self.labels[0]
            gt_label = self.labels[1]
        else:  # xTodo: update here when using new module
            gt_box = self.labels[1]
            gt_label = self.labels[2]
        mbox_locs, mbox_confs, box, box_var = self.env.mid_vars
        loss = fluid.layers.ssd_loss(
            location=mbox_locs,
            confidence=mbox_confs,
            gt_box=gt_box,
            gt_label=gt_label,
            prior_box=box,
            prior_box_var=box_var)
        loss = fluid.layers.reduce_sum(loss)
        loss.persistable = True
        return loss

    def _ssd_feed_list(self, for_export=False):
        # xTodo: update when using new module
        feed_list = [varname for varname in self.base_feed_list]
        if self.is_train_phase:
            feed_list = feed_list[:1] + [label.name for label in self.labels]
        elif self.is_test_phase:
            feed_list = feed_list + [label.name for label in self.labels]
        else:  # self.is_predict_phase:
            if for_export:
                feed_list = [feed_list[0]]
            else:
                # 'image', 'im_id', 'im_shape'
                feed_list = [feed_list[0], self.labels[0].name, feed_list[1]]
        return feed_list

    def _ssd_fetch_list(self, for_export=False):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            # xTodo: update when using new module
            # im_id, bbox, dets, loss
            return [
                self.base_feed_list[1], self.labels[0].name,
                self.outputs[0].name, self.loss.name
            ]
        # im_shape, im_id, bbox
        if for_export:
            return [self.outputs[0].name]
        else:
            return [
                self.base_feed_list[1], self.labels[0].name,
                self.outputs[0].name
            ]

    def _rcnn_build_net(self):
        if self.is_train_phase:
            head_feat = self.feature[0]
        else:
            head_feat = self.predict_feature[0]

        # Rename following layers for: ValueError: Variable cls_score_w has been created before.
        #  the previous shape is (2048, 81); the new shape is (100352, 81).
        #  They are not matched.
        cls_score = fluid.layers.fc(
            input=head_feat,
            size=self.num_classes,
            act=None,
            name='my_cls_score',
            param_attr=ParamAttr(
                name='my_cls_score_w', initializer=Normal(loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name='my_cls_score_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(
            input=head_feat,
            size=4 * self.num_classes,
            act=None,
            name='my_bbox_pred',
            param_attr=ParamAttr(
                name='my_bbox_pred_w', initializer=Normal(loc=0.0,
                                                          scale=0.001)),
            bias_attr=ParamAttr(
                name='my_bbox_pred_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        if self.is_train_phase:
            rpn_cls_loss, rpn_reg_loss, outs = self.feature[1:]
            labels_int32 = outs[1]
            bbox_targets = outs[2]
            bbox_inside_weights = outs[3]
            bbox_outside_weights = outs[4]
            labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
            labels_int64.stop_gradient = True
            loss_cls = fluid.layers.softmax_with_cross_entropy(
                logits=cls_score, label=labels_int64, numeric_stable_mode=True)
            loss_cls = fluid.layers.reduce_mean(loss_cls)
            loss_bbox = fluid.layers.smooth_l1(
                x=bbox_pred,
                y=bbox_targets,
                inside_weight=bbox_inside_weights,
                outside_weight=bbox_outside_weights,
                sigma=1.0)
            loss_bbox = fluid.layers.reduce_mean(loss_bbox)
            total_loss = fluid.layers.sum(
                [loss_bbox, loss_cls, rpn_cls_loss, rpn_reg_loss])
            return [total_loss]
        else:
            rois = self.predict_feature[1]
            im_info = self.base_feed_var_list[1]
            im_shape = self.base_feed_var_list[2]
            im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
            im_scale = fluid.layers.sequence_expand(im_scale, rois)
            boxes = rois / im_scale
            cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
            bbox_pred = fluid.layers.reshape(bbox_pred,
                                             (-1, self.num_classes, 4))
            # decoded_box = self.box_coder(prior_box=boxes, target_box=bbox_pred)
            decoded_box = fluid.layers.box_coder(
                prior_box=boxes,
                prior_box_var=[0.1, 0.1, 0.2, 0.2],
                target_box=bbox_pred,
                code_type='decode_center_size',
                box_normalized=False,
                axis=1)
            cliped_box = fluid.layers.box_clip(
                input=decoded_box, im_info=im_shape)
            # pred_result = self.nms(bboxes=cliped_box, scores=cls_prob)
            pred_result = fluid.layers.multiclass_nms(
                bboxes=cliped_box,
                scores=cls_prob,
                score_threshold=.05,
                nms_top_k=-1,
                keep_top_k=100,
                nms_threshold=.5,
                normalized=False,
                nms_eta=1.0,
                background_label=0)
            if self.is_predict_phase:
                self.env.labels = self._rcnn_add_label()
            return [pred_result]

    def _rcnn_add_label(self):
        if self.is_train_phase:
            idx_list = [
                2,
            ]  # 'im_id'
        elif self.is_test_phase:
            idx_list = [2, 4, 5,
                        6]  # 'im_id', 'gt_box', 'gt_label', 'is_difficult'
        else:  # predict
            idx_list = [2]
        return self._add_label_by_fields(idx_list)

    def _rcnn_add_loss(self):
        if self.is_train_phase:
            loss = self.env.outputs[-1]
        else:
            loss = fluid.layers.fill_constant(
                shape=[1], value=-1, dtype='float32')
        return loss

    def _rcnn_feed_list(self, for_export=False):
        feed_list = [varname for varname in self.base_feed_list]
        if self.is_train_phase:
            # feed_list is ['image', 'im_info', 'gt_box', 'gt_label', 'is_crowd']
            return feed_list[:2] + [self.labels[0].name] + feed_list[2:]
        elif self.is_test_phase:
            # feed list is ['image', 'im_info', 'im_shape']
            return feed_list[:2] + [self.labels[0].name] + feed_list[2:] + \
                   [label.name for label in self.labels[1:]]
        if for_export:
            # skip im_id
            return feed_list[:2] + feed_list[3:]
        else:
            return feed_list[:2] + [self.labels[0].name] + feed_list[2:]

    def _rcnn_fetch_list(self, for_export=False):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            # im_shape, im_id, bbox
            return [
                self.feed_list[2], self.labels[0].name, self.outputs[0].name,
                self.loss.name
            ]

        # im_shape, im_id, bbox
        if for_export:
            return [self.outputs[0].name]
        else:
            return [
                self.feed_list[2], self.labels[0].name, self.outputs[0].name
            ]

    def _yolo_parse_anchors(self, anchors):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors
        """
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _yolo_build_net(self):
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
                   [116, 90], [156, 198], [373, 326]]
        self._yolo_parse_anchors(anchors)

        tip_list = self.feature
        outputs = []
        for i, tip in enumerate(tip_list):
            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                # Rename for: conflict with module pretrain weights
                param_attr=ParamAttr(
                    name="ft_yolo_output.{}.conv.weights".format(i)),
                bias_attr=ParamAttr(
                    regularizer=L2Decay(0.),
                    name="ft_yolo_output.{}.conv.bias".format(i)))
            outputs.append(block_out)

        if self.is_train_phase:
            return outputs

        im_size = self.base_feed_var_list[1]
        boxes = []
        scores = []
        downsample = 32
        for i, output in enumerate(outputs):
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=0.01,
                downsample_ratio=downsample,
                name="yolo_box" + str(i))
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
            downsample //= 2
        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)
        # pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        pred = fluid.layers.multiclass_nms(
            bboxes=yolo_boxes,
            scores=yolo_scores,
            score_threshold=.01,
            nms_top_k=1000,
            keep_top_k=100,
            nms_threshold=0.45,
            normalized=False,
            nms_eta=1.0,
            background_label=-1)
        if self.is_predict_phase:
            self.env.labels = self._yolo_add_label()
        return [pred]

    def _yolo_add_label(self):
        if self.is_train_phase:
            idx_list = [1, 2, 3]  # 'gt_box', 'gt_label', 'gt_score'
        elif self.is_test_phase:
            idx_list = [2, 3, 4,
                        5]  # 'im_id', 'gt_box', 'gt_label', 'is_difficult'
        else:  # predict
            idx_list = [2]
        return self._add_label_by_fields(idx_list)

    def _yolo_add_loss(self):
        if self.is_train_phase:
            gt_box, gt_label, gt_score = self.labels
            outputs = self.outputs
            losses = []
            downsample = 32
            for i, output in enumerate(outputs):
                anchor_mask = self.anchor_masks[i]
                loss = fluid.layers.yolov3_loss(
                    x=output,
                    gt_box=gt_box,
                    gt_label=gt_label,
                    gt_score=gt_score,
                    anchors=self.anchors,
                    anchor_mask=anchor_mask,
                    class_num=self.num_classes,
                    ignore_thresh=0.7,
                    downsample_ratio=downsample,
                    use_label_smooth=True,
                    name="yolo_loss" + str(i))
                losses.append(fluid.layers.reduce_mean(loss))
                downsample //= 2

            loss = sum(losses)
        else:
            loss = fluid.layers.fill_constant(
                shape=[1], value=-1, dtype='float32')
        return loss

    def _yolo_feed_list(self, for_export=False):
        feed_list = [varname for varname in self.base_feed_list]
        if self.is_train_phase:
            return [feed_list[0]] + [label.name for label in self.labels]
        elif self.is_test_phase:
            return feed_list + [label.name for label in self.labels]
        if for_export:
            return feed_list[:2]
        else:
            return feed_list + [self.labels[0].name]

    def _yolo_fetch_list(self, for_export=False):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            # im_shape, im_id, bbox
            return [
                self.feed_list[1], self.labels[0].name, self.outputs[0].name,
                self.loss.name
            ]

        # im_shape, im_id, bbox
        if for_export:
            return [self.outputs[0].name]
        else:
            return [
                self.feed_list[1], self.labels[0].name, self.outputs[0].name
            ]

    def _build_net(self):
        if self.model_type == 'ssd':
            outputs = self._ssd_build_net()
        elif self.model_type == 'rcnn':
            outputs = self._rcnn_build_net()
        elif self.model_type == 'yolo':
            outputs = self._yolo_build_net()
        else:
            raise NotImplementedError
        return outputs

    def _add_label(self):
        if self.model_type == 'ssd':
            labels = self._ssd_add_label()
        elif self.model_type == 'rcnn':
            labels = self._rcnn_add_label()
        elif self.model_type == 'yolo':
            labels = self._yolo_add_label()
        else:
            raise NotImplementedError
        return labels

    def _add_loss(self):
        if self.model_type == 'ssd':
            loss = self._ssd_add_loss()
        elif self.model_type == 'rcnn':
            loss = self._rcnn_add_loss()
        elif self.model_type == 'yolo':
            loss = self._yolo_add_loss()
        else:
            raise NotImplementedError
        return loss

    def _add_metrics(self):
        return []

    @property
    def feed_list(self):
        return self._feed_list(False)

    def _feed_list(self, for_export=False):
        if self.model_type == 'ssd':
            return self._ssd_feed_list(for_export)
        elif self.model_type == 'rcnn':
            return self._rcnn_feed_list(for_export)
        elif self.model_type == 'yolo':
            return self._yolo_feed_list(for_export)
        else:
            raise NotImplementedError

    @property
    def fetch_list(self):
        # ensure fetch loss at last element in train/test phase
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        return self._fetch_list(False)

    def _fetch_list(self, for_export=False):
        if self.model_type == 'ssd':
            return self._ssd_fetch_list(for_export)
        elif self.model_type == 'rcnn':
            return self._rcnn_fetch_list(for_export)
        elif self.model_type == 'yolo':
            return self._yolo_fetch_list(for_export)
        else:
            raise NotImplementedError

    @property
    def fetch_var_list(self):
        fetch_list = self._fetch_list(True)
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in fetch_list]

    @property
    def labels(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.labels

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None):
        with self.phase_guard("predict"):
            fluid.io.save_inference_model(
                dirname=dirname,
                executor=self.exe,
                feeded_var_names=self._feed_list(for_export=True),
                target_vars=self.fetch_var_list,
                main_program=self.main_program,
                model_filename=model_filename,
                params_filename=params_filename)

    def _calculate_metrics(self, run_states):
        loss_sum = run_examples = 0
        run_step = run_time_used = 0
        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(np.array(
                run_state.run_results[-1])) * run_state.run_examples

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        if self.is_train_phase:
            return scores, avg_loss, run_speed

        keys = ['im_shape', 'im_id', 'bbox']
        results = []
        for run_state in run_states:
            outs = [
                run_state.run_results[0], run_state.run_results[1],
                run_state.run_results[2]
            ]
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            results.append(res)

        is_bbox_normalized = dconf.conf[self.model_type]['is_bbox_normalized']
        eval_feed = Feed()
        eval_feed.with_background = dconf.conf[
            self.model_type]['with_background']
        eval_feed.dataset = self.reader

        for metric in self.metrics_choices:
            if metric == "ap":
                box_ap_stats = eval_results(
                    results, eval_feed, 'COCO', self.num_classes, None,
                    is_bbox_normalized, self.config.checkpoint_dir)
                print("box_ap_stats", box_ap_stats)
                scores["ap"] = box_ap_stats[0]
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)

        return scores, avg_loss, run_speed
