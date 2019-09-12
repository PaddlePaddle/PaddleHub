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


class ClassifierTask(BasicTask):
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
            metrics_choices = ["acc"]

        main_program = feature.block.program
        super(ClassifierTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self.feature = feature
        self.num_classes = num_classes
        self.hidden_units = hidden_units

    def _build_net(self):
        cls_feats = self.feature
        if self.hidden_units is not None:
            for n_hidden in self.hidden_units:
                cls_feats = fluid.layers.fc(
                    input=cls_feats, size=n_hidden, act="relu")

        logits = fluid.layers.fc(
            input=cls_feats,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        self.ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(logits, axis=1), shape=[-1, 1])

        return [logits]

    def _add_label(self):
        return [fluid.layers.data(name="label", dtype="int64", shape=[1])]

    def _add_loss(self):
        ce_loss = fluid.layers.cross_entropy(
            input=self.outputs[0], label=self.labels[0])
        return fluid.layers.mean(x=ce_loss)

    def _add_metrics(self):
        acc = fluid.layers.accuracy(input=self.outputs[0], label=self.labels[0])
        return [acc]

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [self.labels[0].name, self.ret_infers.name
                    ] + [metric.name
                         for metric in self.metrics] + [self.loss.name]
        return [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        loss_sum = acc_sum = run_examples = 0
        run_step = run_time_used = 0
        all_labels = np.array([])
        all_infers = np.array([])

        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples
            acc_sum += np.mean(
                run_state.run_results[2]) * run_state.run_examples
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
            if metric == "acc":
                avg_acc = acc_sum / run_examples
                scores["acc"] = avg_acc
            elif metric == "f1":
                f1 = calculate_f1_np(all_infers, all_labels)
                scores["f1"] = f1
            elif metric == "matthews":
                matthews = matthews_corrcoef(all_infers, all_labels)
                scores["matthews"] = matthews
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)

        return scores, avg_loss, run_speed


ImageClassifierTask = ClassifierTask


class TextClassifierTask(ClassifierTask):
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
            metrics_choices = ["acc"]
        super(TextClassifierTask, self).__init__(
            data_reader=data_reader,
            feature=feature,
            num_classes=num_classes,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            hidden_units=hidden_units,
            metrics_choices=metrics_choices)

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
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        self.ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(logits, axis=1), shape=[-1, 1])

        return [logits]


class MultiLabelClassifierTask(ClassifierTask):
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
            metrics_choices = ["auc"]

        main_program = feature.block.program
        super(MultiLabelClassifierTask, self).__init__(
            data_reader=data_reader,
            feature=feature,
            num_classes=num_classes,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            hidden_units=hidden_units,
            metrics_choices=metrics_choices)
        self.class_name = list(data_reader.label_map.keys())

    def _build_net(self):
        cls_feats = fluid.layers.dropout(
            x=self.feature,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        if self.hidden_units is not None:
            for n_hidden in self.hidden_units:
                cls_feats = fluid.layers.fc(
                    input=cls_feats, size=n_hidden, act="relu")

        probs = []
        for i in range(self.num_classes):
            probs.append(
                fluid.layers.fc(
                    input=cls_feats,
                    size=2,
                    param_attr=fluid.ParamAttr(
                        name="cls_out_w_%d" % i,
                        initializer=fluid.initializer.TruncatedNormal(
                            scale=0.02)),
                    bias_attr=fluid.ParamAttr(
                        name="cls_out_b_%d" % i,
                        initializer=fluid.initializer.Constant(0.)),
                    act="softmax"))

        return probs

    def _add_label(self):
        label = fluid.layers.data(
            name="label", shape=[self.num_classes], dtype='int64')
        return [label]

    def _add_loss(self):
        label_split = fluid.layers.split(
            self.labels[0], self.num_classes, dim=-1)
        total_loss = fluid.layers.fill_constant(
            shape=[1], value=0.0, dtype='float64')
        for index, probs in enumerate(self.outputs):
            ce_loss = fluid.layers.cross_entropy(
                input=probs, label=label_split[index])
            total_loss += fluid.layers.reduce_sum(ce_loss)
        loss = fluid.layers.mean(x=total_loss)
        return loss

    def _add_metrics(self):
        label_split = fluid.layers.split(
            self.labels[0], self.num_classes, dim=-1)
        # metrics change to auc of every class
        eval_list = []
        for index, probs in enumerate(self.outputs):
            current_auc, _, _ = fluid.layers.auc(
                input=probs, label=label_split[index])
            eval_list.append(current_auc)
        return eval_list

    def _calculate_metrics(self, run_states):
        loss_sum = acc_sum = run_examples = 0
        run_step = run_time_used = 0
        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples
        auc_list = run_states[-1].run_results[:-1]

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / (run_examples * self.num_classes)
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        for metric in self.metrics_choices:
            if metric == "auc":
                scores["auc"] = np.mean(auc_list)
                # NOTE: for MultiLabelClassifierTask, the metrics will be used to evaluate all the label
                #      and their mean value will also be reported.
                for index, auc in enumerate(auc_list):
                    scores["auc_" + self.class_name[index]] = auc_list[index][0]
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)
        return scores, avg_loss, run_speed

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics] + [self.loss.name]
        return self.outputs
