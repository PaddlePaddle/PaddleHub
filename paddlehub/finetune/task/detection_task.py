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
import paddle.fluid as fluid

from paddlehub.finetune.evaluate import calculate_f1_np, matthews_corrcoef
from .basic_task import BasicTask
from ...contrib.ppdet.utils.eval_utils import eval_results

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

        self.feature = feature
        self.num_classes = num_classes
        self.hidden_units = hidden_units

    @property
    def base_feed_var_list(self):
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in self._base_feed_list]

    def _build_net(self):
        feature_maps = self.feature
        image = self.base_feed_var_list[0]
        conf = {
            'input_size':
            300,
            'aspect_ratios': [[2.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0],
                              [2.0, 3.0], [2.0, 3.0]],
            'min_ratio':
            20,
            'max_ratio':
            90,
            'nms_threshold':
            0.45,
            "nms_top_k":
            400,
            "score_threshold":
            0.01,
            "keep_top_k":
            200,
            "nms_eta":
            1.0,
        }

        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=feature_maps,
            image=image,
            base_size=conf["input_size"],
            num_classes=self.num_classes,
            min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            aspect_ratios=conf["aspect_ratios"],
            min_ratio=conf["min_ratio"],
            max_ratio=conf["max_ratio"])

        self.env.mid_vars = [mbox_locs, mbox_confs, box, box_var]

        nmsed_out = fluid.layers.detection_output(
            mbox_locs,
            mbox_confs,
            box,
            box_var,
            background_label=0,
            nms_threshold=conf["nms_threshold"],
            nms_top_k=conf["nms_top_k"],
            keep_top_k=conf["keep_top_k"],
            score_threshold=conf["score_threshold"],
            nms_eta=conf["nms_eta"])

        self.ret_infers = nmsed_out

        return [nmsed_out]

    def _add_label(self):
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
            fields = ['image', 'gt_box', 'gt_label']
        else:
            fields = [
                'image', 'gt_box', 'gt_label', 'im_shape', 'im_id',
                'is_difficult'
            ]

        labels = []
        for key in fields[1:]:
            l = fluid.layers.data(
                name=feed_var_map[key]['name'],
                shape=feed_var_map[key]['shape'],
                dtype=feed_var_map[key]['dtype'],
                lod_level=feed_var_map[key]['lod_level'])
            labels.append(l)
        return labels

    @property
    def return_numpy(self):
        return 2  # return lod tensor

    def _add_loss(self):
        # ce_loss = fluid.layers.cross_entropy(
        #     input=self.outputs[0], label=self.labels[0])
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

    def _add_metrics(self):
        # acc = fluid.layers.accuracy(input=self.outputs[0], label=self.labels[0])
        return []

    @property
    def fetch_list(self):
        # Todo:
        if self.is_train_phase:
            return [
                self.labels[0].name, self.labels[1].name, self.ret_infers.name,
                self.loss.name
            ]
        elif self.is_test_phase:
            return [
                self.labels[2].name, self.labels[3].name, self.ret_infers.name,
                self.loss.name
            ]
        return [output.name for output in self.outputs]

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

        is_bbox_normalized = True
        eval_feed = Feed()
        eval_feed.with_background = True
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
