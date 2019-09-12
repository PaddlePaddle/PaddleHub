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
from paddlehub.finetune.evaluate import chunk_eval, calculate_f1
from .basic_task import BasicTask


class SequenceLabelTask(BasicTask):
    def __init__(self,
                 feature,
                 max_seq_len,
                 num_classes,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 metrics_choices="default"):
        if metrics_choices == "default":
            metrics_choices = ["f1", "precision", "recall"]

        main_program = feature.block.program
        super(SequenceLabelTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)
        self.feature = feature
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

    def _build_net(self):
        self.logits = fluid.layers.fc(
            input=self.feature,
            size=self.num_classes,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_seq_label_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_seq_label_out_b",
                initializer=fluid.initializer.Constant(0.)))

        self.ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(self.logits, axis=2), shape=[-1, 1])
        ret_infers = fluid.layers.assign(self.ret_infers)

        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64')
        seq_len = fluid.layers.assign(self.seq_len)

        logits = self.logits
        logits = fluid.layers.flatten(logits, axis=2)
        logits = fluid.layers.softmax(logits)
        self.num_labels = logits.shape[1]
        return [logits]

    def _add_label(self):
        label = fluid.layers.data(
            name="label", shape=[self.max_seq_len, 1], dtype='int64')
        return [label]

    def _add_loss(self):
        labels = fluid.layers.flatten(self.labels[0], axis=2)
        ce_loss = fluid.layers.cross_entropy(
            input=self.outputs[0], label=labels)
        loss = fluid.layers.mean(x=ce_loss)
        return loss

    def _add_metrics(self):
        self.ret_labels = fluid.layers.reshape(x=self.labels[0], shape=[-1, 1])
        return [self.ret_labels, self.ret_infers, self.seq_len]

    def _calculate_metrics(self, run_states):
        total_infer = total_label = total_correct = loss_sum = 0
        run_step = run_time_used = run_examples = 0
        for run_state in run_states:
            loss_sum += np.mean(run_state.run_results[-1])
            np_labels = run_state.run_results[0]
            np_infers = run_state.run_results[1]
            np_lens = run_state.run_results[2]
            label_num, infer_num, correct_num = chunk_eval(
                np_labels, np_infers, np_lens, self.num_labels,
                self.device_count)
            total_infer += infer_num
            total_label += label_num
            total_correct += correct_num
            run_examples += run_state.run_examples
            run_step += run_state.run_step

        run_time_used = time.time() - run_states[0].run_time_begin
        run_speed = run_step / run_time_used
        avg_loss = loss_sum / run_examples

        precision, recall, f1 = calculate_f1(total_label, total_infer,
                                             total_correct)
        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()

        for metric in self.metrics_choices:
            if metric == "precision":
                scores["precision"] = precision
            elif metric == "recall":
                scores["recall"] = recall
            elif metric == "f1":
                scores["f1"] = f1
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)

        return scores, avg_loss, run_speed

    @property
    def feed_list(self):
        feed_list = [varname for varname in self._base_feed_list]
        if self.is_train_phase or self.is_test_phase:
            feed_list += [self.labels[0].name, self.seq_len.name]
        else:
            feed_list += [self.seq_len.name]
        return feed_list

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics] + [self.loss.name]
        elif self.is_predict_phase:
            return [self.ret_infers.name] + [self.seq_len.name]
        return [output.name for output in self.outputs]
