# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json

import paddle.fluid as fluid

from chinese_electra_base.model.transformer_encoder import encoder, pre_process_layer


class ElectraConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing electra model config file '%s'" % config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ElectraModel(object):
    def __init__(self, src_ids, position_ids, sentence_ids, input_mask, config, weight_sharing=True, use_fp16=False):

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(scale=config['initializer_range'])

        self._build_model(src_ids, position_ids, sentence_ids, input_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids, input_mask):
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(name=self._pos_emb_name, initializer=self._param_initializer))

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(name=self._sent_emb_name, initializer=self._param_initializer))

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        emb_out = pre_process_layer(emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder')

        if self._dtype == "float16":
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)

        self_attn_mask = fluid.layers.matmul(x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name='encoder')

    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(input=self._enc_out, axes=[1], starts=[0], ends=[1])
        return next_sent_feat

    def get_pretraining_output(self, mask_label, mask_pos, labels):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(name='mask_lm_trans_fc.w_0', initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        # transform: layer norm
        mask_trans_feat = pre_process_layer(mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0", initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size], dtype=self._dtype, attr=mask_lm_out_bias_attr, is_bias=True)

        else:
            fc_out = fluid.layers.fc(
                input=mask_trans_feat,
                size=self._voc_size,
                param_attr=fluid.ParamAttr(name="mask_lm_out_fc.w_0", initializer=self._param_initializer),
                bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(logits=fc_out, label=mask_label)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            size=2,
            param_attr=fluid.ParamAttr(name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")

        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.mean(next_sent_loss)

        loss = mean_next_sent_loss + mean_mask_lm_loss
        return next_sent_acc, mean_mask_lm_loss, loss
