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

from .basic_task import BasicTask
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


class DetectionTask(BasicTask):
    def __init__(self,
                 feature,
                 num_classes,
                 feed_list,
                 data_reader,
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
        self.model_type = 'ssd'

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

    def _inner_add_label(self, start_index):
        feed_var_map = {var['name']: var for var in feed_var_def}
        # tensor padding with 0 is used instead of LoD tensor when
        # num_max_boxes is set
        if getattr(self, 'num_max_boxes', None) is not None:
            feed_var_map['gt_label']['shape'] = [self.num_max_boxes]
            feed_var_map['gt_score']['shape'] = [self.num_max_boxes]
            feed_var_map['gt_box']['shape'] = [self.num_max_boxes, 4]
            feed_var_map['is_difficult']['shape'] = [self.num_max_boxes]
            feed_var_map['gt_label']['lod_level'] = 0
            feed_var_map['gt_score']['lod_level'] = 0
            feed_var_map['gt_box']['lod_level'] = 0
            feed_var_map['is_difficult']['lod_level'] = 0

        # Todo: 必须和下面loss时取label顺序一致；必须与Arrange后field顺序一致
        if self.is_train_phase:
            fields = dconf.feed_config[self.model_type]['train']['fields']
        elif self.is_test_phase:
            fields = dconf.feed_config[self.model_type]['dev']['fields']
        else:  # Cannot go to here
            raise RuntimeError("Cannot go to _add_label in predict phase")
            # fields = dconf.feed_config[self.model_type]['predict']['fields']

        labels = []
        for key in fields[start_index:]:
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

        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=feature_list,
            image=image,
            base_size=300,
            num_classes=self.num_classes,
            min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            aspect_ratios=[[2.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0],
                              [2.0, 3.0], [2.0, 3.0]],
            min_ratio=20,
            max_ratio=90)

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
        return [nmsed_out]

    def _ssd_add_label(self):
        # train: 'gt_box', 'gt_label'
        # dev: 'gt_box', 'gt_label', 'im_shape', 'im_id', 'is_difficult'
        start_index = 1
        return self._inner_add_label(start_index)

    def _ssd_add_loss(self):
        gt_box = self.labels[0]
        gt_label = self.labels[1]
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

    def _ssd_fetch_list(self):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            return [
                self.labels[2].name, self.labels[3].name, self.outputs[0].name,
                self.loss.name]
        return [output.name for output in self.outputs]

    def _rcnn_build_net(self):
        if self.is_train_phase:
            head_feat = self.feature[0]
        else:
            head_feat = self.predict_feature[0]

        cls_score = fluid.layers.fc(input=head_feat,
                                    size=self.num_classes,
                                    act=None,
                                    name='cls_score',
                                    param_attr=ParamAttr(
                                        name='cls_score_w',
                                        initializer=Normal(
                                            loc=0.0, scale=0.01)),
                                    bias_attr=ParamAttr(
                                        name='cls_score_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(input=head_feat,
                                    size=4 * self.num_classes,
                                    act=None,
                                    name='bbox_pred',
                                    param_attr=ParamAttr(
                                        name='bbox_pred_w',
                                        initializer=Normal(
                                            loc=0.0, scale=0.001)),
                                    bias_attr=ParamAttr(
                                        name='bbox_pred_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))

        if self.is_train_phase:
            rpn_cls_loss, rpn_reg_loss = self.feature[-2:]
            outs = self.feature[:5]
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
            total_loss = fluid.layers.sum([loss_bbox, loss_cls, rpn_cls_loss, rpn_reg_loss])
            return [total_loss]
        else:
            rois = self.predict_feature[1]
            im_info = self.base_feed_var_list[1]
            im_shape = self.base_feed_var_list[2]
            im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
            im_scale = fluid.layers.sequence_expand(im_scale, rois)
            boxes = rois / im_scale
            cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
            bbox_pred = fluid.layers.reshape(bbox_pred, (-1, self.num_classes, 4))
            # decoded_box = self.box_coder(prior_box=boxes, target_box=bbox_pred)
            decoded_box = fluid.layers.box_coder(
                prior_box=boxes, prior_box_var=[0.1, 0.1, 0.2, 0.2],
                target_box=bbox_pred, code_type='decode_center_size',
                box_normalized=False, axis=1)
            cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=im_shape)
            # pred_result = self.nms(bboxes=cliped_box, scores=cls_prob)
            pred_result = fluid.layers.multiclass_nms(
                bboxes=cliped_box, scores=cls_prob,
                score_threshold=.05,
                nms_top_k=-1,
                keep_top_k=100,
                nms_threshold=.5,
                normalized=False,
                nms_eta=1.0,
                background_label=0
            )
            return [pred_result]

    def _rcnn_add_label(self):
        if self.is_train_phase:
            start_index = -1  # 'im_id'
        else:  # test
            start_index = 3  # 'gt_box', 'gt_label', 'is_difficult', 'im_id'
        return self._inner_add_label(start_index)

    def _rcnn_add_loss(self):
        if self.is_train_phase:
            loss = self.env.outputs[-1]
        else:
            loss = fluid.layers.fill_constant(shape=[], value=-1, dtype='float32')
        return loss

    def _rcnn_fetch_list(self):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            return [self.feed_list[2], self.labels[-1].name, self.outputs[0].name, self.loss.name]
        return [self.outputs[0].name]

    def _build_net(self):
        if self.model_type == 'ssd':
            outputs = self._ssd_build_net()
        elif self.model_type == 'rcnn':
            outputs = self._rcnn_build_net()
        else:
            raise NotImplementedError
        return outputs

    def _add_label(self):
        if self.model_type == 'ssd':
            labels = self._ssd_add_label()
        elif self.model_type == 'rcnn':
            labels = self._rcnn_add_label()
        else:
            raise NotImplementedError
        return labels

    @property
    def return_numpy(self):
        return 2  # return lod tensor

    def _add_loss(self):
        if self.model_type == 'ssd':
            loss = self._ssd_add_loss()
        elif self.model_type == 'rcnn':
            loss = self._rcnn_add_loss()
        else:
            raise NotImplementedError
        return loss

    def _add_metrics(self):
        return []

    @property
    def feed_list(self):
        feed_list = [varname for varname in self.base_feed_list]
        if self.is_train_phase or self.is_test_phase:
            feed_list += [label.name for label in self.labels]
        return feed_list

    @property
    def fetch_list(self):
        # ensure fetch loss at last element in train/test phase
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.model_type == 'ssd':
            return self._ssd_fetch_list()
        elif self.model_type == 'rcnn':
            return self._rcnn_fetch_list()
        else:
            raise NotImplementedError

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
        eval_feed.with_background = dconf.conf[self.model_type]['with_background']
        eval_feed.dataset = self.reader

        for metric in self.metrics_choices:
            if metric == "ap":
                box_ap_stats = eval_results(results, eval_feed, 'COCO',
                                            self.num_classes, None,
                                            is_bbox_normalized, None, None)
                print("box_ap_stats", box_ap_stats)
                scores["ap"] = box_ap_stats[0]
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)

        return scores, avg_loss, run_speed
