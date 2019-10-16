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
from scipy.stats import spearmanr
from .basic_task import BasicTask


class RegressionTask(BasicTask):
    def __init__(self,
                 feature,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 hidden_units=None,
                 metrics_choices="default"):
        if metrics_choices == "default":
            metrics_choices = ["spearman"]

        main_program = feature.block.program
        super(RegressionTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)
        self.feature = feature
        self.hidden_units = hidden_units

    def _build_net(self):
        cls_feats = fluid.layers.dropout(
            x=self.feature,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        if self.hidden_units is not None:
            for n_hidden in self.hidden_units:
                cls_feats = fluid.layers.fc(
                    input=cls_feats, size=n_hidden, act="relu")

        logits = fluid.layers.fc(
            input=cls_feats,
            size=1,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act=None)

        return [logits]

    def _add_label(self):
        return [fluid.layers.data(name="label", dtype="float32", shape=[1])]

    def _add_loss(self):
        cost = fluid.layers.square_error_cost(
            input=self.outputs[0], label=self.labels[0])
        return fluid.layers.mean(x=cost)

    def _add_metrics(self):
        return []

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [self.labels[0].name, self.outputs[0].name
                    ] + [metric.name
                         for metric in self.metrics] + [self.loss.name]
        return [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        loss_sum = run_examples = 0
        run_step = run_time_used = 0
        all_labels = np.array([])
        all_infers = np.array([])
        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples
            np_labels = run_state.run_results[0]
            np_infers = run_state.run_results[1]
            all_labels = np.hstack((all_labels, np_labels.reshape([-1])))
            all_infers = np.hstack((all_infers, np_infers.reshape([-1])))

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()

        for metric in self.metrics_choices:
            if metric == "spearman":
                spearman_correlations = spearmanr(all_labels, all_infers)[0]
                scores["spearman"] = spearman_correlations
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)
        return scores, avg_loss, run_speed
