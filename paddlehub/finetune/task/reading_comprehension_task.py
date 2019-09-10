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
from .basic_task import BasicTask


class ReadingComprehensionTask(BasicTask):
    def __init__(self,
                 feature,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 metrics_choices=None):

        main_program = feature.block.program
        super(ReadingComprehensionTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)
        self.feature = feature

    def _build_net(self):
        if self.is_predict_phase:
            self.unique_id = fluid.layers.data(
                name="start_positions",
                shape=[-1, 1],
                lod_level=0,
                dtype="int64")

        logits = fluid.layers.fc(
            input=self.feature,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_seq_label_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_seq_label_out_b",
                initializer=fluid.initializer.Constant(0.)))

        logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
        start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

        batch_ones = fluid.layers.fill_constant_batch_size_like(
            input=start_logits, dtype='int64', shape=[1], value=1)
        num_seqs = fluid.layers.reduce_sum(input=batch_ones)

        return [start_logits, end_logits, num_seqs]

    def _add_label(self):
        start_positions = fluid.layers.data(
            name="start_positions", shape=[-1, 1], lod_level=0, dtype="int64")
        end_positions = fluid.layers.data(
            name="end_positions", shape=[-1, 1], lod_level=0, dtype="int64")
        return [start_positions, end_positions]

    def _add_loss(self):
        start_positions = self.labels[0]
        end_positions = self.labels[1]

        start_logits = self.outputs[0]
        end_logits = self.outputs[1]

        start_loss = fluid.layers.softmax_with_cross_entropy(
            logits=start_logits, label=start_positions)
        start_loss = fluid.layers.mean(x=start_loss)
        end_loss = fluid.layers.softmax_with_cross_entropy(
            logits=end_logits, label=end_positions)
        end_loss = fluid.layers.mean(x=end_loss)
        total_loss = (start_loss + end_loss) / 2.0
        return total_loss

    def _add_metrics(self):
        return []

    @property
    def feed_list(self):
        feed_list = [varname for varname in self._base_feed_list]
        if self.is_train_phase:
            feed_list += [self.labels[0].name, self.labels[1].name]
        elif self.is_predict_phase:
            feed_list += [self.unique_id.name]
        return feed_list

    @property
    def fetch_list(self):
        if self.is_train_phase:
            return [metric.name for metric in self.metrics
                    ] + [self.loss.name, self.outputs[-1].name]
        elif self.is_predict_phase:
            return [self.unique_id.name
                    ] + [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        total_cost, total_num_seqs = [], []
        run_step = run_time_used = run_examples = 0
        for run_state in run_states:
            np_loss = run_state.run_results[0]
            np_num_seqs = run_state.run_results[1]
            total_cost.extend(np_loss * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
            run_examples += run_state.run_examples
            run_step += run_state.run_step

        run_time_used = time.time() - run_states[0].run_time_begin
        run_speed = run_step / run_time_used
        avg_loss = np.sum(total_cost) / np.sum(total_num_seqs)

        scores = OrderedDict()
        # If none of metrics has been implemented, loss will be used to evaluate.
        return scores, avg_loss, run_speed
