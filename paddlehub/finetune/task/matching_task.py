# coding:utf-8
#  Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from paddlehub.common.logger import logger
from paddlehub.finetune.evaluate import calculate_f1_np, simple_accuracy
from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
from paddlehub.tokenizer.tokenizer import CustomTokenizer
import numpy as np
import paddle.fluid as fluid
import paddlehub.network as net

from .base_task import BaseTask


class PairwiseTextMatchingTask(BaseTask):
    """
    Create a pairwise text macthing task, which needs 3 slots (query, left_title, right_title).
    """

    def __init__(self,
                 query_feature,
                 left_feature,
                 right_feature,
                 tokenizer,
                 dataset=None,
                 network=None,
                 config=None,
                 metrics_choices=["acc", "f1", "precision", "recall"]):
        """
        Args:
            query_feature(Variable): It represents the query in the text matching task.
            left_feature(Variable): It represents the left title in the text matching task.
            right_feature(Variable): If the task is pair_wise, then the right_feature must be set and it represents the right title. It is optional.
            tokenizer(object): tokenizer(object): It should be hub.BertTokenizer or hub.CustomTokenizer, which tokenizes the text and encodes the data as model needed.
            dataset(object): The text macthing dataset.
            network(str): The pre-defined network. Choices: 'bow', 'cnn', 'gru' and 'lstm'. Default None.
            config (RunConfig): run config for the task, such as batch_size, epoch, learning_rate setting and so on.
            metrics_choices(list): metrics used to the task, default ['acc', 'f1', "precision", "recall"].
        """

        if network:
            assert network in [
                'bow', 'cnn', 'gru', 'lstm'
            ], 'network (%s) choice must be one of bow, cnn, gru, lstm!' % network
            assert len(query_feature.shape) == 3 and len(
                left_feature.shape
            ) == 3 and (
                len(right_feature.shape) == 3
            ), 'When you use network, the parameter query_feature and left_feature must be the token-level feature ([batch_size, max_seq_len, embedding_size]), shape as [-1, 128, 200].'

        self.tokenizer = tokenizer
        self.query_feature = query_feature
        self.left_feature = left_feature
        self.right_feature = right_feature
        self.network = network

        main_program = query_feature.block.program
        super(PairwiseTextMatchingTask, self).__init__(
            dataset=dataset,
            data_reader=None,
            main_program=main_program,
            startup_program=None,
            config=config,
            metrics_choices=metrics_choices)

    def _build_net(self):
        if self.network:
            self.seq_len_1 = fluid.layers.data(
                name="seq_len", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_1_used = fluid.layers.squeeze(self.seq_len_1, axes=[1])

            self.seq_len_2 = fluid.layers.data(
                name="seq_len_2", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_2_used = fluid.layers.squeeze(self.seq_len_2, axes=[1])

            self.seq_len_3 = fluid.layers.data(
                name="seq_len_3", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_3_used = fluid.layers.squeeze(self.seq_len_3, axes=[1])

            # unpad the token_feature
            query_unpad = fluid.layers.sequence_unpad(
                self.query_feature, length=self.seq_len_1_used)
            left_unpad = fluid.layers.sequence_unpad(
                self.left_feature, length=self.seq_len_2_used)
            right_unpad = fluid.layers.sequence_unpad(
                self.right_feature, length=self.seq_len_3_used)

            # add pre-defined net
            net_func = getattr(net.matching, self.network)
            if self.is_train_phase or self.is_predict_phase:
                logger.info("%s has been added in the PairwiseTextMatchingTask!"
                            % self.network)

            query_feats, left_feats = net_func(query_unpad, left_unpad)
            query_feats = fluid.layers.fc(
                input=query_feats,
                size=300,
                param_attr=fluid.ParamAttr(
                    name="query_fc_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="query_fc_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")
            left_feats = fluid.layers.fc(
                input=left_feats,
                size=300,
                param_attr=fluid.ParamAttr(
                    name="title_fc_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="title_fc_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")

            left_concat = fluid.layers.concat(
                input=[query_feats, left_feats], axis=1)

            _, right_feats = net_func(query_unpad, right_unpad)
            right_feats = fluid.layers.fc(
                input=right_feats,
                size=300,
                param_attr=fluid.ParamAttr(
                    name="title_fc_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="title_fc_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")

            right_concat = fluid.layers.concat(
                input=[query_feats, right_feats], axis=1)

        else:
            query_feats = fluid.layers.dropout(
                x=self.query_feature,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            query_mean = fluid.layers.reduce_mean(query_feats, dim=1)

            left_feats = fluid.layers.dropout(
                x=self.left_feature,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            left_mean = fluid.layers.reduce_mean(left_feats, dim=1)

            left_sub = fluid.layers.elementwise_sub(query_mean, left_mean)
            left_sub = fluid.layers.abs(left_sub)

            left_concat = fluid.layers.concat(
                input=[query_mean, left_mean, left_sub], axis=-1)

            right_feats = fluid.layers.dropout(
                x=self.right_feature,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            right_mean = fluid.layers.reduce_mean(right_feats, dim=1)

            right_sub = fluid.layers.elementwise_sub(query_mean, right_mean)
            right_sub = fluid.layers.abs(right_sub)

            right_concat = fluid.layers.concat(
                input=[query_mean, right_mean, right_sub],
                axis=-1,
            )

        left_score = fluid.layers.fc(
            input=left_concat,
            size=1,
            param_attr=fluid.ParamAttr(
                name="matching_out_w_left",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02),
            ),
            bias_attr=fluid.ParamAttr(
                name="matching_out_b_left",
                initializer=fluid.initializer.Constant(0.),
            ),
        )
        left_score = fluid.layers.sigmoid(left_score)

        right_score = fluid.layers.fc(
            input=right_concat,
            size=1,
            param_attr=fluid.ParamAttr(
                name="matching_out_w_right",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02),
            ),
            bias_attr=fluid.ParamAttr(
                name="matching_out_b_right",
                initializer=fluid.initializer.Constant(0.),
            ),
        )
        right_score = fluid.layers.sigmoid(right_score)

        return [left_score, right_score]

    def _add_label(self):
        labels = fluid.layers.data(
            name="label", dtype="int64", shape=[-1, 1], lod_level=0)
        labels = fluid.layers.cast(x=labels, dtype="float32")
        return [labels]

    def _add_loss(self):
        rank_loss = fluid.layers.rank_loss(
            label=self.labels[0], left=self.outputs[0], right=self.outputs[1])
        return fluid.layers.mean(x=rank_loss)

    def _add_metrics(self):
        return self.outputs

    @property
    def feed_list(self):
        if self.is_train_phase or self.is_test_phase:
            feed_list = super(PairwiseTextMatchingTask, self).feed_list
        else:
            if isinstance(self.tokenizer, CustomTokenizer):
                feed_list = ['text', 'text_2', 'text_3']
                if self.network:
                    feed_list += ['seq_len', 'seq_len_2', 'seq_len_3']
            elif isinstance(self.tokenizer, BertTokenizer):
                feed_list = [
                    'input_ids', 'segment_ids', 'input_mask', 'position_ids',
                    'input_ids_2', 'segment_ids_2', 'input_mask_2',
                    'position_ids_2', 'input_ids_3', 'segment_ids_3',
                    'input_mask_3', 'position_ids_3'
                ]
                if self.network:
                    feed_list += ['seq_len', 'seq_len_2', 'seq_len_3']
            else:
                raise RuntimeError(
                    "Unknown Tokenizer %s." % self.tokenizer.__class__)
        return feed_list

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics
                    ] + [self.labels[0].name, self.loss.name]
        return [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        loss_sum = run_examples = 0
        run_step = run_time_used = 0
        all_labels = []
        all_infers = []

        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples

            left_scores, right_scores, labels = run_state.run_results[:-1]
            for index in range(left_scores.shape[0]):
                if left_scores[index] > right_scores[index]:
                    prediction = 1
                else:
                    prediction = 0
                all_infers.append(prediction)
                all_labels.append(int(labels[index][0]))

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        precision, recall, f1 = calculate_f1_np(all_infers, all_labels)
        acc = simple_accuracy(all_infers, all_labels)
        for metric in self.metrics_choices:
            if metric == "precision":
                scores["precision"] = precision
            elif metric == "recall":
                scores["recall"] = recall
            elif metric == "f1":
                scores["f1"] = f1
            elif metric == "acc":
                scores["acc"] = acc
            else:
                raise ValueError(
                    "Unknown metric: %s! The chosen metrics must be acc, f1, presicion or recall."
                    % metric)

        return scores, avg_loss, run_speed

    def _encode_matching_data(self, data, max_seq_len):
        encoded_data = []
        for text_pair in data:
            record = {}
            record_a = self.tokenizer.encode(
                text=text_pair[0], max_seq_len=max_seq_len)
            record_b = self.tokenizer.encode(
                text=text_pair[1], max_seq_len=max_seq_len)
            if isinstance(self.tokenizer, BertTokenizer):
                record = {
                    # text_1
                    'input_ids': record_a['input_ids'],
                    'segment_ids': record_a['segment_ids'],
                    'input_mask': record_a['input_mask'],
                    'position_ids': record_a['position_ids'],
                    'seq_len': record_a['seq_len'],
                    # text_2
                    'input_ids_2': record_b['input_ids'],
                    'segment_ids_2': record_b['segment_ids'],
                    'input_mask_2': record_b['input_mask'],
                    'position_ids_2': record_b['position_ids'],
                    'seq_len_2': record_b['seq_len'],
                    # text_3 same as text_2
                    'input_ids_3': record_b['input_ids'],
                    'segment_ids_3': record_b['segment_ids'],
                    'input_mask_3': record_b['input_mask'],
                    'position_ids_3': record_b['position_ids'],
                    'seq_len_3': record_b['seq_len'],
                }

            elif isinstance(self.tokenizer, CustomTokenizer):
                record = {
                    # text_1
                    'text': record_a['text'],
                    'seq_len': record_a['seq_len'],
                    # text_2
                    'text_2': record_b['text'],
                    'seq_len_2': record_b['seq_len'],
                    # text_3 same as text_3
                    'text_3': record_b['text'],
                    'seq_len_3': record_b['seq_len'],
                }
            else:
                raise Exception(
                    "Unknown Tokenizer %s!" % self.tokenizer.__class__)

            encoded_data.append(record)

        return encoded_data

    def predict(self,
                data,
                max_seq_len,
                label_list=None,
                load_best_model=True,
                return_result=False,
                accelerate_mode=True):
        """
        make prediction for the input data.

        Args:
            data (list): The data will be predicted. It should be a text pair, such as [[query, title], ...].
            max_seq_len(int): It will limit the total sequence returned so that it has a maximum length.
            label_list(list): The labels which was exploited as fine-tuning. If the return_result as true, the label_list must be set.
            load_best_model (bool): Whether loading the best model or not.
            return_result (bool): Whether getting the final result (such as predicted labels) or not. Default as True.
            accelerate_mode (bool): use high-performance predictor or not
        Returns:
            RunState(object): The running result in the predict phase, which includes the fetch results as running program.
        """
        self.accelerate_mode = accelerate_mode

        encoded_data = self._encode_matching_data(data, max_seq_len)

        with self.phase_guard(phase="predict"):
            self._predict_data = encoded_data
            self._label_list = label_list
            self._predict_start_event()

            if load_best_model:
                self.init_if_load_best_model()
            else:
                self.init_if_necessary()
            if not self.accelerate_mode:
                run_states = self._run()
            else:
                if not self._predictor:
                    self._predictor = self._create_predictor()
                run_states = self._run_with_predictor()
            self._predict_end_event(run_states)
            self._predict_data = None
            if return_result or not self._compatible_mode:
                return self._postprocessing(run_states)

        return run_states

    def _postprocessing(self, run_states):
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            left_scores, right_scores = batch_results[0], batch_results[1]
            for index in range(left_scores.shape[0]):
                if left_scores[index] > right_scores[index]:
                    prediction = 1
                else:
                    prediction = 0
                results.append(prediction)

        return results


class PointwiseTextMatchingTask(BaseTask):
    """
    Create a  pointwise text macthing task, which needs 2 slots (query, title).
    """

    def __init__(self,
                 query_feature,
                 title_feature,
                 tokenizer,
                 dataset=None,
                 network=None,
                 config=None,
                 metrics_choices=["acc", 'f1', "precision", "recall"]):
        """
        Args:
            query_feature(Variable): It represents the query in the text matching task.
            title_feature(Variable): It represents the title in the text matching task.
            tokenizer(object): tokenizer(object): It should be hub.BertTokenizer or hub.CustomTokenizer, which tokenizes the text and encodes the data as model needed.
            dataset(object): The text macthing dataset.
            network(str): The pre-defined network. Choices: 'bow', 'cnn', 'gru' and 'lstm'. Default None.
            config (RunConfig): run config for the task, such as batch_size, epoch, learning_rate setting and so on.
            metrics_choices(list): metrics used to the task, default ['acc', 'f1', "precision", "recall"].
        """

        if network:
            assert network in [
                'bow', 'cnn', 'gru', 'lstm'
            ], 'network (%s) choice must be one of bow, cnn, gru, lstm!' % network
            assert len(query_feature.shape) == 3 and len(
                title_feature.shape
            ) == 3, 'When you use network, the parameter query_feature and title_feature must be the token-level feature ([batch_size, max_seq_len, embedding_size]), shape as [-1, 128, 200].'

        self.tokenizer = tokenizer
        self.query_feature = query_feature
        self.title_feature = title_feature
        self.network = network

        main_program = query_feature.block.program
        super(PointwiseTextMatchingTask, self).__init__(
            dataset=dataset,
            data_reader=None,
            main_program=main_program,
            startup_program=None,
            config=config,
            metrics_choices=metrics_choices)

    def _build_net(self):
        if self.network:
            self.seq_len_1 = fluid.layers.data(
                name="seq_len", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_1_used = fluid.layers.squeeze(self.seq_len_1, axes=[1])

            self.seq_len_2 = fluid.layers.data(
                name="seq_len_2", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_2_used = fluid.layers.squeeze(self.seq_len_2, axes=[1])

            # unpad the token_feature
            query_unpad = fluid.layers.sequence_unpad(
                self.query_feature, length=self.seq_len_1_used)
            title_unpad = fluid.layers.sequence_unpad(
                self.title_feature, length=self.seq_len_2_used)

            # add pre-defined net
            net_func = getattr(net.matching, self.network)
            if self.is_train_phase or self.is_predict_phase:
                logger.info(
                    "%s has been added in the PointwiseTextMatchingTask!" %
                    self.network)

            query_feats, title_feats = net_func(query_unpad, title_unpad)

            query_fc = fluid.layers.fc(
                input=query_feats,
                size=300,
                param_attr=fluid.ParamAttr(
                    name="query_fc_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="query_fc_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")
            title_fc = fluid.layers.fc(
                input=title_feats,
                size=300,
                param_attr=fluid.ParamAttr(
                    name="title_fc_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="title_fc_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")
            title_concat = fluid.layers.concat(
                input=[query_fc, title_fc], axis=1)

            fc1 = fluid.layers.fc(
                input=title_concat,
                size=256,
                param_attr=fluid.ParamAttr(
                    name="matching_fc1_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="matching_fc1_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")
            fc2 = fluid.layers.fc(
                input=fc1,
                size=128,
                param_attr=fluid.ParamAttr(
                    name="matching_fc2_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="matching_fc2_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")
            projection = fluid.layers.fc(
                input=fc2,
                size=96,
                param_attr=fluid.ParamAttr(
                    name="matching_fc3_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                ),
                bias_attr=fluid.ParamAttr(
                    name="matching_fc3_b",
                    initializer=fluid.initializer.Constant(0.),
                ),
                act="tanh")

        else:
            query_feats = fluid.layers.dropout(
                x=self.query_feature,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            query_mean = fluid.layers.reduce_mean(query_feats, dim=1)
            title_feats = fluid.layers.dropout(
                x=self.title_feature,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            title_mean = fluid.layers.reduce_mean(title_feats, dim=1)

            sub = fluid.layers.elementwise_sub(query_mean, title_mean)
            sub = fluid.layers.abs(sub)

            projection = fluid.layers.concat(
                input=[query_mean, title_mean, sub], axis=-1)

        score = fluid.layers.fc(
            input=projection,
            size=2,
            param_attr=fluid.ParamAttr(
                name="matching_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02),
            ),
            bias_attr=fluid.ParamAttr(
                name="matching_out_b",
                initializer=fluid.initializer.Constant(0.),
            ),
            act="softmax")

        return [score]

    def _add_label(self):
        return [fluid.layers.data(name="label", dtype="int64", shape=[-1, 1])]

    def _add_loss(self):
        ce_loss = fluid.layers.cross_entropy(
            input=self.outputs[0], label=self.labels[0])
        return fluid.layers.mean(x=ce_loss)

    def _add_metrics(self):
        return self.outputs

    @property
    def feed_list(self):
        if self.is_train_phase or self.is_test_phase:
            feed_list = super(PointwiseTextMatchingTask, self).feed_list
        else:
            if isinstance(self.tokenizer, CustomTokenizer):
                feed_list = ['text', 'text_2']
                if self.network:
                    feed_list += ['seq_len', 'seq_len_2']
            elif isinstance(self.tokenizer, BertTokenizer):
                feed_list = [
                    'input_ids', 'segment_ids', 'input_mask', 'position_ids',
                    'input_ids_2', 'segment_ids_2', 'input_mask_2',
                    'position_ids_2'
                ]
                if self.network:
                    feed_list += ['seq_len', 'seq_len_2']
            else:
                raise RuntimeError(
                    "Unknown Tokenizer %s." % self.tokenizer.__class__)
        return feed_list

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics
                    ] + [self.labels[0].name, self.loss.name]
        return [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        loss_sum = run_examples = 0
        run_step = run_time_used = 0
        all_labels = []
        all_infers = []

        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples

            scores, labels = run_state.run_results[:-1]

            predictions = np.argmax(scores, axis=1)

            all_infers += [int(pre) for pre in predictions]
            true_labels = [int(label[0]) for label in labels]
            all_labels += true_labels

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        precision, recall, f1 = calculate_f1_np(all_infers, all_labels)
        acc = simple_accuracy(all_infers, all_labels)
        for metric in self.metrics_choices:
            if metric == "precision":
                scores["precision"] = precision
            elif metric == "recall":
                scores["recall"] = recall
            elif metric == "f1":
                scores["f1"] = f1
            elif metric == "acc":
                scores["acc"] = acc
            else:
                raise ValueError(
                    "Unknown metric: %s! The chosen metrics must be acc, f1, presicion or recall."
                    % self.metrics_choice)

        return scores, avg_loss, run_speed

    def _encode_matching_data(self, data, max_seq_len):
        encoded_data = []
        for text_pair in data:
            record = {}
            record_a = self.tokenizer.encode(
                text=text_pair[0], max_seq_len=max_seq_len)
            record_b = self.tokenizer.encode(
                text=text_pair[1], max_seq_len=max_seq_len)
            if isinstance(self.tokenizer, BertTokenizer):
                record = {
                    # text_1
                    'input_ids': record_a['input_ids'],
                    'segment_ids': record_a['segment_ids'],
                    'input_mask': record_a['input_mask'],
                    'position_ids': record_a['position_ids'],
                    'seq_len': record_a['seq_len'],
                    # text_2
                    'input_ids_2': record_b['input_ids'],
                    'segment_ids_2': record_b['segment_ids'],
                    'input_mask_2': record_b['input_mask'],
                    'position_ids_2': record_b['position_ids'],
                    'seq_len_2': record_b['seq_len'],
                }
            elif isinstance(self.tokenizer, CustomTokenizer):
                record = {
                    # text_1
                    'text': record_a['text'],
                    'seq_len': record_a['seq_len'],
                    # text_2
                    'text_2': record_b['text'],
                    'seq_len_2': record_b['seq_len'],
                }
            else:
                raise Exception(
                    "Unknown Tokenizer %s!" % self.tokenizer.__class__)

            encoded_data.append(record)

        return encoded_data

    def predict(self,
                data,
                max_seq_len,
                label_list=None,
                load_best_model=True,
                return_result=False,
                accelerate_mode=True):
        """
        make prediction for the input data.

        Args:
            data (list): The data will be predicted. It should be a text pair, such as [[query, title], ...].
            max_seq_len(int): It will limit the total sequence returned so that it has a maximum length.
            label_list(list): The labels which was exploited as fine-tuning. If the return_result as true, the label_list must be set.
            load_best_model (bool): Whether loading the best model or not.
            return_result (bool): Whether getting the final result (such as predicted labels) or not. Default as True.
            accelerate_mode (bool): use high-performance predictor or not
        Returns:
            RunState(object): The running result in the predict phase, which includes the fetch results as running program.
        """
        self.accelerate_mode = accelerate_mode

        encoded_data = self._encode_matching_data(data, max_seq_len)

        with self.phase_guard(phase="predict"):
            self._predict_data = encoded_data
            self._label_list = label_list
            self._predict_start_event()

            if load_best_model:
                self.init_if_load_best_model()
            else:
                self.init_if_necessary()
            if not self.accelerate_mode:
                run_states = self._run()
            else:
                if not self._predictor:
                    self._predictor = self._create_predictor()
                run_states = self._run_with_predictor()
            self._predict_end_event(run_states)
            self._predict_data = None
            if return_result or not self._compatible_mode:
                return self._postprocessing(run_states)
        return run_states

    def _postprocessing(self, run_states):
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infer = np.argmax(batch_results[0], axis=1)
            results += [
                self._label_list[sample_infer] for sample_infer in batch_infer
            ]
        return results
