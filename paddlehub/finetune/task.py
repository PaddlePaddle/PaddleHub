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

import os
import collections
import time
import multiprocessing

import numpy as np
import paddle.fluid as fluid


class Task(object):
    """
    A simple transfer learning task definition,
    including Paddle's main_program, startup_program and inference program
    """

    def __init__(self,
                 task_type,
                 graph_var_dict,
                 main_program,
                 startup_program,
                 inference_program=None):
        self.task_type = task_type
        self.graph_var_dict = graph_var_dict
        self._main_program = main_program
        self._startup_program = startup_program
        self._inference_program = inference_program
        self._test_program = main_program.clone(for_test=True)

    def variable(self, var_name):
        if var_name in self.graph_var_dict:
            return self.graph_var_dict[var_name]

        raise KeyError("var_name {} not in task graph".format(var_name))

    def main_program(self):
        return self._main_program

    def startup_program(self):
        return self._startup_program

    def inference_program(self):
        return self._inference_program

    def test_program(self):
        return self._test_program

    def metric_variable_names(self):
        metric_variable_names = []
        for var_name in self.graph_var_dict:
            metric_variable_names.append(var_name)

        return metric_variable_names


def create_text_cls_task(feature, num_classes, hidden_units=None):
    """
    Append a multi-layer perceptron classifier for binary classification base
    on input feature
    """
    program = feature.block.program
    with fluid.program_guard(program):
        cls_feats = fluid.layers.dropout(
            x=feature,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        # append fully connected layer according to hidden_units
        if hidden_units is not None:
            for n_hidden in hidden_units:
                cls_feats = fluid.layers.fc(input=cls_feats, size=n_hidden)

        logits = fluid.layers.fc(
            input=cls_feats,
            size=num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        inference_program = fluid.default_main_program().clone(for_test=True)
        label = fluid.layers.data(name="label", dtype="int64", shape=[1])
        ce_loss = fluid.layers.cross_entropy(input=logits, label=label)
        loss = fluid.layers.mean(x=ce_loss)

        num_example = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=logits, label=label, total=num_example)

        graph_var_dict = {
            "loss": loss,
            "accuracy": accuracy,
            "num_example": num_example,
            "label": label,
            "probs": logits
        }

        task = Task(
            "text_classification",
            graph_var_dict,
            fluid.default_main_program(),
            fluid.default_startup_program(),
            inference_program=inference_program)

    return task


def create_img_cls_task(feature, num_classes, hidden_units=None):
    """
    Create the transfer learning task for image classification.
    Args:
        feature:

    Return:
        Task

    Raise:
        None
    """
    program = feature.block.program
    with fluid.program_guard(program):
        cls_feats = feature
        # append fully connected layer according to hidden_units
        if hidden_units is not None:
            for n_hidden in hidden_units:
                cls_feats = fluid.layers.fc(input=cls_feats, size=n_hidden)

        probs = fluid.layers.fc(
            input=cls_feats,
            size=num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        inference_program = fluid.default_main_program().clone(for_test=True)

        label = fluid.layers.data(name="label", dtype="int64", shape=[1])
        ce_loss = fluid.layers.cross_entropy(input=probs, label=label)
        loss = fluid.layers.mean(x=ce_loss)

        num_example = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=probs, label=label, total=num_example)

        graph_var_dict = {
            "loss": loss,
            "probs": probs,
            "accuracy": accuracy,
            "num_example": num_example,
            "label": label,
            "probs": probs
        }

        task = Task(
            "image_classification",
            graph_var_dict,
            fluid.default_main_program(),
            fluid.default_startup_program(),
            inference_program=inference_program)

    return task


def create_seq_label_task(feature, max_seq_len, num_classes):
    program = feature.block.program
    with fluid.program_guard(program):
        logits = fluid.layers.fc(
            input=feature,
            size=num_classes,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_seq_label_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_seq_label_out_b",
                initializer=fluid.initializer.Constant(0.)))

        ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(logits, axis=2), shape=[-1, 1])

        logits = fluid.layers.flatten(logits, axis=2)
        logits = fluid.layers.softmax(logits)

        inference_program = fluid.default_main_program().clone(for_test=True)

        seq_len = fluid.layers.data(name="seq_len", shape=[1], dtype='int64')
        label = fluid.layers.data(
            name="label", shape=[max_seq_len, 1], dtype='int64')
        ret_labels = fluid.layers.reshape(x=label, shape=[-1, 1])

        labels = fluid.layers.flatten(label, axis=2)
        ce_loss = fluid.layers.cross_entropy(input=logits, label=labels)
        loss = fluid.layers.mean(x=ce_loss)

        graph_var_dict = {
            "loss": loss,
            "probs": logits,
            "labels": ret_labels,
            "infers": ret_infers,
            "seq_len": seq_len,
            "label": label
        }

        task = Task(
            "sequence_labeling",
            graph_var_dict,
            fluid.default_main_program(),
            fluid.default_startup_program(),
            inference_program=inference_program)

    return task
