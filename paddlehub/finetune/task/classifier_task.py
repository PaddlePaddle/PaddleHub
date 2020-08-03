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
from collections import OrderedDict
import numpy as np
import paddle.fluid as fluid
import time

from paddlehub.common.logger import logger
from paddlehub.finetune.evaluate import calculate_f1_np, matthews_corrcoef
from paddlehub.reader.nlp_reader import ClassifyReader, LACClassifyReader

import paddlehub.network as net

from .base_task import BaseTask


class ClassifierTask(BaseTask):
    def __init__(self,
                 feature,
                 num_classes,
                 dataset=None,
                 feed_list=None,
                 data_reader=None,
                 startup_program=None,
                 config=None,
                 hidden_units=None,
                 metrics_choices="default"):
        if metrics_choices == "default":
            metrics_choices = ["acc"]

        main_program = feature.block.program
        super(ClassifierTask, self).__init__(
            dataset=dataset,
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
        precision, recall, f1 = calculate_f1_np(all_infers, all_labels)
        matthews = matthews_corrcoef(all_infers, all_labels)
        for metric in self.metrics_choices:
            if metric == "precision":
                scores["precision"] = precision
            elif metric == "recall":
                scores["recall"] = recall
            elif metric == "f1":
                scores["f1"] = f1
            elif metric == "acc":
                scores["acc"] = acc_sum / run_examples
            elif metric == "matthews":
                scores["matthews"] = matthews
            else:
                raise ValueError(
                    "Unknown metric: %s! The chosen metrics must be acc, f1, presicion or recall."
                    % metric)

        return scores, avg_loss, run_speed

    def _postprocessing(self, run_states):
        if self._compatible_mode:
            try:
                label_list = list(self._base_data_reader.label_map.keys())
            except:
                raise Exception(
                    "ImageClassificationDataset does not support postprocessing, please use BaseCVDataset instead"
                )
        else:
            if self._label_list:
                label_list = self._label_list
            else:
                logger.warning(
                    "Fail to postprocess the predict output. Please set label_list parameter in predict function or initialize the task with dataset parameter."
                )
                return run_states
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results
            batch_infer = np.argmax(batch_result[0], axis=1)
            results += [
                label_list[sample_infer] for sample_infer in batch_infer
            ]
        return results


ImageClassifierTask = ClassifierTask


class TextClassifierTask(ClassifierTask):
    """
    Create a text classification task.
    It will use full-connect layer with softmax activation function to classify texts.
    """

    def __init__(
            self,
            num_classes,
            dataset=None,
            feed_list=None,  # Deprecated
            data_reader=None,  # Deprecated
            feature=None,
            token_feature=None,
            network=None,
            startup_program=None,
            config=None,
            hidden_units=None,
            metrics_choices="default"):
        """
        Args:
            num_classes: total labels of the text classification task.
            feed_list(list): the variable name that will be feeded to the main program, Deprecated in paddlehub v1.8.
            data_reader(object): data reader for the task. It must be one of ClassifyReader and LACClassifyReader, Deprecated in paddlehub v1.8..
            feature(Variable): the `feature` will be used to classify texts. It must be the sentence-level feature, shape as [-1, emb_size]. `Token_feature` and `feature` couldn't be setted at the same time. One of them must be setted as not None. Default None.
            token_feature(Variable): the `feature` will be used to connect the pre-defined network. It must be the token-level feature, shape as [-1, seq_len, emb_size]. Default None.
            network(str): the pre-defined network. Choices: 'bilstm', 'bow', 'cnn', 'dpcnn', 'gru' and 'lstm'. Default None. If network is setted, then `token_feature` must be setted and `feature` must be None.
            startup_program (object): the customized startup program, default None.
            config (RunConfig): run config for the task, such as batch_size, epoch, learning_rate setting and so on. Default None.
            hidden_units(list): the element of `hidden_units` list is the full-connect layer size. It will add the full-connect layers to the program. Default None.
            metrics_choices(list): metrics used to the task, default ["acc"]. Choices: acc, precision, recall, f1, matthews.
        """
        if (not feature) and (not token_feature):
            logger.error(
                'Both token_feature and feature are None, one of them must be set.'
            )
            exit(1)
        elif feature and token_feature:
            logger.error(
                'Both token_feature and feature are set. One should be set, the other should be None.'
            )
            exit(1)

        if network:
            assert network in [
                'bilstm', 'bow', 'cnn', 'dpcnn', 'gru', 'lstm'
            ], 'network (%s) choice must be one of bilstm, bow, cnn, dpcnn, gru, lstm!' % network
            assert token_feature and (
                not feature
            ), 'If you wanna use network, you must set token_feature ranther than feature for TextClassifierTask!'
            assert len(
                token_feature.shape
            ) == 3, 'When you use network, the parameter token_feature must be the token-level feature([batch_size, max_seq_len, embedding_size]), shape as [-1, 128, 200].'
        else:
            assert feature and (
                not token_feature
            ), 'If you do not use network, you must set feature ranther than token_feature for TextClassifierTask!'
            assert len(
                feature.shape
            ) == 2, 'When you do not use network, the parameter feture must be the sentence-level feature ([batch_size, hidden_size]), such as the pooled_output of ERNIE, BERT, RoBERTa and ELECTRA module.'

        self.network = network

        if metrics_choices == "default":
            metrics_choices = ["acc"]

        super(TextClassifierTask, self).__init__(
            dataset=dataset,
            data_reader=data_reader,
            feature=feature if feature else token_feature,
            num_classes=num_classes,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            hidden_units=hidden_units,
            metrics_choices=metrics_choices)

    def _build_net(self):
        if not isinstance(self._base_data_reader, LACClassifyReader):
            # LACClassifyReader won't return the seqence length, while Dataset with tokenizer and ClassifyReader will.
            self.seq_len = fluid.layers.data(
                name="seq_len", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_used = fluid.layers.squeeze(self.seq_len, axes=[1])
            # unpad the token_feature
            unpad_feature = fluid.layers.sequence_unpad(
                self.feature, length=self.seq_len_used)
        if self.network:
            # add pre-defined net
            net_func = getattr(net.classification, self.network)
            if self.network == 'dpcnn':
                # deepcnn network is no need to unpad
                cls_feats = net_func(
                    self.feature, emb_dim=self.feature.shape[-1])
            else:
                if self._compatible_mode and isinstance(self._base_data_reader,
                                                        LACClassifyReader):
                    cls_feats = net_func(self.feature)
                else:
                    cls_feats = net_func(unpad_feature)
            if self.is_train_phase or self.is_predict_phase:
                logger.info("%s has been added in the TextClassifierTask!" %
                            self.network)
        else:
            # not use pre-defined net but to use fc net
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

    @property
    def feed_list(self):
        if self._compatible_mode:
            feed_list = [varname for varname in self._base_feed_list]
            if isinstance(self._base_data_reader, ClassifyReader):
                # ClassifyReader will return the seqence length of an input text
                feed_list += [self.seq_len.name]
            if self.is_train_phase or self.is_test_phase:
                feed_list += [self.labels[0].name]
        else:
            feed_list = super(TextClassifierTask, self).feed_list
        return feed_list

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            fetch_list = [
                self.labels[0].name, self.ret_infers.name, self.metrics[0].name,
                self.loss.name
            ]
        else:
            # predict phase
            if isinstance(self._base_data_reader, LACClassifyReader):
                fetch_list = [self.outputs[0].name]
            else:
                fetch_list = [self.outputs[0].name, self.seq_len.name]

        return fetch_list


class MultiLabelClassifierTask(ClassifierTask):
    def __init__(self,
                 feature,
                 num_classes,
                 dataset=None,
                 feed_list=None,
                 data_reader=None,
                 startup_program=None,
                 config=None,
                 hidden_units=None,
                 metrics_choices="default"):
        if metrics_choices == "default":
            metrics_choices = ["auc"]

        super(MultiLabelClassifierTask, self).__init__(
            dataset=dataset,
            data_reader=data_reader,
            feature=feature,
            num_classes=num_classes,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            hidden_units=hidden_units,
            metrics_choices=metrics_choices)
        if self._compatible_mode:
            self.class_name = list(data_reader.label_map.keys())
        else:
            self.class_name = self._label_list

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
        return [output.name for output in self.outputs]

    def _postprocessing(self, run_states):
        results = []
        if self._compatible_mode:
            label_list = list(self._base_data_reader.label_map.keys())
        else:
            if self._label_list:
                label_list = self._label_list
            else:
                logger.warning(
                    "Fail to postprocess the predict output. Please set label_list parameter in predict function or initialize the task with dataset parameter."
                )
                return run_states

        for batch_state in run_states:
            batch_result = batch_state.run_results
            for sample_id in range(len(batch_result[0])):
                sample_result = []
                for category_id in range(len(label_list)):
                    sample_category_prob = batch_result[category_id][sample_id]
                    sample_category_value = np.argmax(sample_category_prob)
                    sample_result.append(
                        {label_list[category_id]: sample_category_value})
                results.append(sample_result)
        return results
