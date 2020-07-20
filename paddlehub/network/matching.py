# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
This module provide nets for text classification
"""
import paddle.fluid as fluid
import paddle.fluid.param_attr as attr


def bow(left_emb, right_emb, hid_dim=128):
    """
    BOW network.
    """
    # bow layer

    left_pool = fluid.layers.sequence_pool(input=left_emb, pool_type='sum')
    right_pool = fluid.layers.sequence_pool(input=right_emb, pool_type='sum')
    left_soft = fluid.layers.softsign(left_pool)
    right_soft = fluid.layers.softsign(right_pool)
    return left_soft, right_soft


def cnn(left_emb, right_emb, hid_dim=128, filter_size=3, num_filters=256):
    """
    CNN network.
    """
    # cnn layer
    left_cnn = fluid.nets.sequence_conv_pool(
        input=left_emb,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=attr.ParamAttr(name="matching_cnn"),
        act="relu",
        pool_type="max")
    right_cnn = fluid.nets.sequence_conv_pool(
        input=right_emb,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=attr.ParamAttr(name="matching_cnn"),
        act="relu",
        pool_type="max")
    return left_cnn, right_cnn


def gru(left_emb, right_emb, hid_dim=128):
    """
    GRU network.
    """
    left_proj = fluid.layers.fc(
        input=left_emb,
        size=hid_dim * 3,
        param_attr=attr.ParamAttr(name="matching_gru_fc.w"),
        bias_attr=attr.ParamAttr(name="matching_gru_fc.b"))
    right_proj = fluid.layers.fc(
        input=right_emb,
        size=hid_dim * 3,
        param_attr=attr.ParamAttr(name="matching_gru_fc.w"),
        bias_attr=attr.ParamAttr(name="matching_gru_fc.b"))
    left_gru = fluid.layers.dynamic_gru(
        input=left_proj,
        size=hid_dim,
        param_attr=attr.ParamAttr(name="matching_gru.w"),
        bias_attr=attr.ParamAttr(name="matching_gru.b"))
    right_gru = fluid.layers.dynamic_gru(
        input=right_proj,
        size=hid_dim,
        param_attr=attr.ParamAttr(name="matching_gru.w"),
        bias_attr=attr.ParamAttr(name="matching_gru.b"))
    left_last = fluid.layers.sequence_last_step(left_gru)
    right_last = fluid.layers.sequence_last_step(right_gru)
    return left_last, right_last


def lstm(left_emb, right_emb, hid_dim=128):
    """
    LSTM network.
    """
    left_proj = fluid.layers.fc(
        input=left_emb,
        size=hid_dim * 4,
        param_attr=attr.ParamAttr(name="matching_lstm_fc.w"),
        bias_attr=attr.ParamAttr(name="matching_lstm_fc.b"))
    right_proj = fluid.layers.fc(
        input=right_emb,
        size=hid_dim * 4,
        param_attr=attr.ParamAttr(name="matching_lstm_fc.w"),
        bias_attr=attr.ParamAttr(name="matching_lstm_fc.b"))
    left_lstm, cell = fluid.layers.dynamic_lstm(
        input=left_proj,
        size=hid_dim * 4,
        param_attr=attr.ParamAttr(name="matching_lstm.w"),
        bias_attr=attr.ParamAttr(name="matching_lstm.b"),
        is_reverse=False)
    right_lstm, _ = fluid.layers.dynamic_lstm(
        input=right_proj,
        size=hid_dim * 4,
        param_attr=attr.ParamAttr(name="matching_lstm.w"),
        bias_attr=attr.ParamAttr(name="matching_lstm.b"),
        is_reverse=False)
    left_last = fluid.layers.sequence_last_step(left_lstm)
    right_last = fluid.layers.sequence_last_step(right_lstm)
    return left_last, right_last
