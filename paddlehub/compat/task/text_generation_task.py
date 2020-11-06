# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import time
from collections import OrderedDict

import numpy as np
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.layers import RNNCell, LSTMCell, rnn, BeamSearchDecoder, dynamic_decode

from paddlehub.compat.task.metrics import compute_bleu
from paddlehub.compat.task.base_task import BaseTask


class AttentionDecoderCell(RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0., init_scale=0.1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []
        self.init_scale = init_scale
        param_attr = ParamAttr(initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale))
        bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0))
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, param_attr, bias_attr))

    def attention(self, query, enc_output, mask=None):
        query = fluid.layers.unsqueeze(query, [1])
        memory = fluid.layers.fc(
            enc_output,
            self.hidden_size,
            num_flatten_dims=2,
            param_attr=ParamAttr(
                name='dec_memory_w',
                initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale)))
        attn = fluid.layers.matmul(query, memory, transpose_y=True)

        if mask:
            attn = fluid.layers.transpose(attn, [1, 0, 2])
            attn = fluid.layers.elementwise_add(attn, mask * 1000000000, -1)
            attn = fluid.layers.transpose(attn, [1, 0, 2])
        weight = fluid.layers.softmax(attn)
        weight_memory = fluid.layers.matmul(weight, memory)

        return weight_memory

    def call(self, step_input, states, enc_output, enc_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = fluid.layers.concat([step_input, input_feed], 1)
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, lstm_states[i])
            step_input = fluid.layers.dropout(
                out, self.dropout_prob, dropout_implementation='upscale_in_train') if self.dropout_prob > 0 else out
            new_lstm_states.append(new_lstm_state)
        dec_att = self.attention(step_input, enc_output, enc_padding_mask)
        dec_att = fluid.layers.squeeze(dec_att, [1])
        concat_att_out = fluid.layers.concat([dec_att, step_input], 1)
        out = fluid.layers.fc(
            concat_att_out,
            self.hidden_size,
            param_attr=ParamAttr(
                name='dec_out_w',
                initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale)))
        return out, [new_lstm_states, out]


class TextGenerationTask(BaseTask):
    '''
    TextGenerationTask use rnn as decoder and beam search technology when predict.
    Args:
        feature(Variable): The sentence-level feature, shape as [-1, emb_size].
        token_feature(Variable): The token-level feature, shape as [-1, seq_len, emb_size].
        max_seq_len(int): the decoder max sequence length.
        num_classes(int): total labels of the task.
        dataset(GenerationDataset): the dataset containing training set, development set and so on. If you want to finetune the model, you should set it.
                 Otherwise, if you just want to use the model to predict, you can omit it. Default None
        num_layers(int): the decoder rnn layers number. Default 1
        hidden_size(int): the decoder rnn hidden size. Default 128
        dropout(float): the decoder dropout rate. Default 0.
        beam_size(int): the beam search size during predict phase. Default 10.
        beam_max_step_num(int): the beam search max step number. Default 30.
        start_token(str): the beam search start token. Default '<s>'
        end_token(str): the beam search end token. Default '</s>'
        startup_program(Program): the customized startup_program, default None
        config(RunConfig): the config for the task, default None
        metrics_choices(list): metrics used to the task, default ['bleu']
    '''

    def __init__(
            self,
            feature,
            token_feature,
            max_seq_len,
            num_classes,
            dataset=None,
            num_layers=1,
            hidden_size=512,
            dropout=0.,
            beam_size=10,
            beam_max_step_num=30,
            start_token='<s>',
            end_token='</s>',
            startup_program=None,
            config=None,
            metrics_choices='default',
    ):
        if metrics_choices == 'default':
            metrics_choices = ['bleu']
        main_program = feature.block.program
        super(TextGenerationTask, self).__init__(
            dataset=dataset,
            main_program=main_program,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.token_feature = token_feature
        self.feature = feature
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.beam_size = beam_size
        self.beam_max_step_num = beam_max_step_num
        self.start_token = start_token
        self.end_token = end_token

    def _add_label(self):
        label = fluid.layers.data(name='label', shape=[self.max_seq_len, 1], dtype='int64')
        return [label]

    def _build_net(self):
        self.seq_len = fluid.layers.data(name='seq_len', shape=[1], dtype='int64', lod_level=0)
        self.seq_len_used = fluid.layers.squeeze(self.seq_len, axes=[1])
        src_mask = fluid.layers.sequence_mask(self.seq_len_used, maxlen=self.max_seq_len, dtype='float32')
        enc_padding_mask = (src_mask - 1.0)

        # Define decoder and initialize it.
        dec_cell = AttentionDecoderCell(self.num_layers, self.hidden_size, self.dropout)
        dec_init_hidden = fluid.layers.fc(
            input=self.feature,
            size=self.hidden_size,
            num_flatten_dims=1,
            param_attr=fluid.ParamAttr(
                name='dec_init_hidden_w', initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(name='dec_init_hidden_b', initializer=fluid.initializer.Constant(0.)))
        dec_initial_states = [
            [[dec_init_hidden,
              dec_cell.get_initial_states(batch_ref=self.feature, shape=[self.hidden_size])]] * self.num_layers,
            dec_cell.get_initial_states(batch_ref=self.feature, shape=[self.hidden_size])
        ]
        tar_vocab_size = len(self._label_list)
        tar_embeder = lambda x: fluid.embedding(
            input=x,
            size=[tar_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='target_embedding', initializer=fluid.initializer.UniformInitializer(low=-0.1, high=0.1)))
        start_token_id = self._label_list.index(self.start_token)
        end_token_id = self._label_list.index(self.end_token)
        if not self.is_predict_phase:
            self.dec_input = fluid.layers.data(name='dec_input', shape=[self.max_seq_len], dtype='int64')
            tar_emb = tar_embeder(self.dec_input)
            dec_output, _ = rnn(
                cell=dec_cell,
                inputs=tar_emb,
                initial_states=dec_initial_states,
                sequence_length=None,
                enc_output=self.token_feature,
                enc_padding_mask=enc_padding_mask)
            self.logits = fluid.layers.fc(
                dec_output,
                size=tar_vocab_size,
                num_flatten_dims=len(dec_output.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name='output_w', initializer=fluid.initializer.UniformInitializer(low=-0.1, high=0.1)))
            self.ret_infers = fluid.layers.reshape(x=fluid.layers.argmax(self.logits, axis=2), shape=[-1, 1])
            logits = self.logits
            logits = fluid.layers.softmax(logits)
            return [logits]
        else:
            output_layer = lambda x: fluid.layers.fc(
                x, size=tar_vocab_size, num_flatten_dims=len(x.shape) - 1, param_attr=fluid.ParamAttr(name='output_w'))
            beam_search_decoder = BeamSearchDecoder(
                dec_cell,
                start_token_id,
                end_token_id,
                self.beam_size,
                embedding_fn=tar_embeder,
                output_fn=output_layer)
            enc_output = beam_search_decoder.tile_beam_merge_with_batch(self.token_feature, self.beam_size)
            enc_padding_mask = beam_search_decoder.tile_beam_merge_with_batch(enc_padding_mask, self.beam_size)
            self.ret_infers, _ = dynamic_decode(
                beam_search_decoder,
                inits=dec_initial_states,
                max_step_num=self.beam_max_step_num,
                enc_output=enc_output,
                enc_padding_mask=enc_padding_mask)
            return self.ret_infers

    def _postprocessing(self, run_states):
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infers = batch_results[0].astype(np.int32)
            seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
            for i, sample_infers in enumerate(batch_infers):
                beam_result = []
                for beam_infer in sample_infers.T:
                    seq_result = [self._label_list[infer] for infer in beam_infer.tolist()[:seq_lens[i] - 2]]
                    beam_result.append(seq_result)
                results.append(beam_result)
        return results

    def _add_metrics(self):
        self.ret_labels = fluid.layers.reshape(x=self.labels[0], shape=[-1, 1])
        return [self.ret_labels, self.ret_infers, self.seq_len_used]

    def _add_loss(self):
        loss = fluid.layers.cross_entropy(input=self.outputs[0], label=self.labels[0], soft_label=False)
        loss = fluid.layers.unsqueeze(loss, axes=[2])
        max_tar_seq_len = fluid.layers.shape(self.dec_input)[1]
        tar_sequence_length = fluid.layers.elementwise_sub(self.seq_len_used, fluid.layers.ones_like(self.seq_len_used))
        tar_mask = fluid.layers.sequence_mask(tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        loss = loss * tar_mask
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        return loss

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics] + [self.loss.name]
        elif self.is_predict_phase:
            return [self.ret_infers.name] + [self.seq_len_used.name]
        return [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        loss_sum = 0
        run_step = run_examples = 0
        labels = []
        results = []
        for run_state in run_states:
            loss_sum += np.mean(run_state.run_results[-1])
            np_labels = run_state.run_results[0]
            np_infers = run_state.run_results[1]
            np_lens = run_state.run_results[2]
            batch_size = len(np_lens)
            max_len = len(np_labels) // batch_size
            for i in range(batch_size):
                label = [
                    self.dataset.label_list[int(id)] for id in np_labels[i * max_len:i * max_len + np_lens[i] - 2]
                ]  # -2 for CLS and SEP
                result = [
                    self.dataset.label_list[int(id)] for id in np_infers[i * max_len:i * max_len + np_lens[i] - 2]
                ]
                labels.append(label)
                results.append(result)

            run_examples += run_state.run_examples
            run_step += run_state.run_step

        run_time_used = time.time() - run_states[0].run_time_begin
        run_speed = run_step / run_time_used
        avg_loss = loss_sum / run_examples

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        for metric in self.metrics_choices:
            if metric == 'bleu':
                scores['bleu'] = compute_bleu(labels, results, max_order=1)[0]
            else:
                raise ValueError('Not Support Metric: \'%s\'' % metric)
        return scores, avg_loss, run_speed
