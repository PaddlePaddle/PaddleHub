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
                 metrics_choices="default",
                 add_crf=False):
        if metrics_choices == "default":
            metrics_choices = ["f1", "precision", "recall"]

        self.add_crf = add_crf

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

    @property
    def return_numpy(self):
        if self.add_crf:
            return False
        else:
            return True

    def _build_net(self):
        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64')
        seq_len = fluid.layers.assign(self.seq_len)

        if self.add_crf:
            unpad_feature = fluid.layers.sequence_unpad(
                self.feature, length=self.seq_len)
            self.emission = fluid.layers.fc(
                size=self.num_classes,
                input=unpad_feature,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(low=-0.1, high=0.1),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            size = self.emission.shape[1]
            fluid.layers.create_parameter(
                shape=[size + 2, size], dtype=self.emission.dtype, name='crfw')
            self.ret_infers = fluid.layers.crf_decoding(
                input=self.emission, param_attr=fluid.ParamAttr(name='crfw'))
            ret_infers = fluid.layers.assign(self.ret_infers)
            return [ret_infers]
        else:
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
        if self.add_crf:
            labels = fluid.layers.sequence_unpad(self.labels[0], self.seq_len)
            crf_cost = fluid.layers.linear_chain_crf(
                input=self.emission,
                label=labels,
                param_attr=fluid.ParamAttr(name='crfw'))
            loss = fluid.layers.mean(x=crf_cost)
        else:
            labels = fluid.layers.flatten(self.labels[0], axis=2)
            ce_loss = fluid.layers.cross_entropy(
                input=self.outputs[0], label=labels)
            loss = fluid.layers.mean(x=ce_loss)
        return loss

    def _add_metrics(self):
        if self.add_crf:
            labels = fluid.layers.sequence_unpad(self.labels[0], self.seq_len)
            (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
             num_correct_chunks) = fluid.layers.chunk_eval(
                 input=self.outputs[0],
                 label=labels,
                 chunk_scheme="IOB",
                 num_chunk_types=int(np.ceil((self.num_classes - 1) / 2.0)))
            chunk_evaluator = fluid.metrics.ChunkEvaluator()
            chunk_evaluator.reset()
            return [precision, recall, f1_score]
        else:
            self.ret_labels = fluid.layers.reshape(
                x=self.labels[0], shape=[-1, 1])
            return [self.ret_labels, self.ret_infers, self.seq_len]

    def _calculate_metrics(self, run_states):
        total_infer = total_label = total_correct = loss_sum = 0
        run_step = run_time_used = run_examples = 0
        precision_sum = recall_sum = f1_score_sum = 0
        for run_state in run_states:
            loss_sum += np.mean(run_state.run_results[-1])
            if self.add_crf:
                precision_sum += np.mean(
                    run_state.run_results[0]) * run_state.run_examples
                recall_sum += np.mean(
                    run_state.run_results[1]) * run_state.run_examples
                f1_score_sum += np.mean(
                    run_state.run_results[2]) * run_state.run_examples
            else:
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

        if self.add_crf:
            precision = precision_sum / run_examples
            recall = recall_sum / run_examples
            f1 = f1_score_sum / run_examples
        else:
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
