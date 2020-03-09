#coding:utf-8
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import paddlehub as hub
import paddle.fluid as fluid
from paddlehub import logger


class _BERTEmbeddingTask(hub.BaseTask):
    def __init__(self,
                 pooled_feature,
                 seq_feature,
                 feed_list,
                 data_reader,
                 config=None):
        main_program = pooled_feature.block.program
        super(_BERTEmbeddingTask, self).__init__(
            main_program=main_program,
            data_reader=data_reader,
            feed_list=feed_list,
            config=config,
            metrics_choices=[])
        self.pooled_feature = pooled_feature
        self.seq_feature = seq_feature

    def _build_net(self):
        return [self.pooled_feature, self.seq_feature]

    def _postprocessing(self, run_states):
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results
            batch_pooled_features = batch_result[0]
            batch_seq_features = batch_result[1]
            for i in range(len(batch_pooled_features)):
                results.append(
                    [batch_pooled_features[i], batch_seq_features[i]])
        return results


class BERTModule(hub.Module):
    def _initialize(self):
        """
        Must override this method.

        some member variables are required, others are optional.
        """
        # required config
        self.ernie_config = None
        self.MAX_SEQ_LEN = None
        self.params_path = None
        self.vocab_path = None
        # optional config
        self.spm_path = None
        self.word_dict_path = None
        raise NotImplementedError

    def init_pretraining_params(self, exe, pretraining_params_path,
                                main_program):
        assert os.path.exists(
            pretraining_params_path
        ), "[%s] cann't be found." % pretraining_params_path

        def existed_params(var):
            if not isinstance(var, fluid.framework.Parameter):
                return False
            return os.path.exists(
                os.path.join(pretraining_params_path, var.name))

        fluid.io.load_vars(
            exe,
            pretraining_params_path,
            main_program=main_program,
            predicate=existed_params)
        logger.info("Load pretraining parameters from {}.".format(
            pretraining_params_path))

    def context(
            self,
            max_seq_len=128,
            trainable=True,
    ):
        """
        get inputs, outputs and program from pre-trained module

        Args:
            max_seq_len (int): the max sequence length
            trainable (bool): optimizing the pre-trained module params during training or not

        Returns: inputs, outputs, program.
                 The inputs is a dict with keys named input_ids, position_ids, segment_ids, input_mask and task_ids
                 The outputs is a dict with two keys named pooled_output and sequence_output.

        """

        assert max_seq_len <= self.MAX_SEQ_LEN and max_seq_len >= 1, "max_seq_len({}) should be in the range of [1, {}]".format(
            max_seq_len, self.MAX_SEQ_LEN)

        module_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(module_program, startup_program):
            with fluid.unique_name.guard("@HUB_%s@" % self.name):
                input_ids = fluid.layers.data(
                    name='input_ids',
                    shape=[-1, max_seq_len, 1],
                    dtype='int64',
                    lod_level=0)
                position_ids = fluid.layers.data(
                    name='position_ids',
                    shape=[-1, max_seq_len, 1],
                    dtype='int64',
                    lod_level=0)
                segment_ids = fluid.layers.data(
                    name='segment_ids',
                    shape=[-1, max_seq_len, 1],
                    dtype='int64',
                    lod_level=0)
                input_mask = fluid.layers.data(
                    name='input_mask',
                    shape=[-1, max_seq_len, 1],
                    dtype='float32',
                    lod_level=0)
                pooled_output, sequence_output = self.net(
                    input_ids, position_ids, segment_ids, input_mask)

        inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
        }

        outputs = {
            "pooled_output": pooled_output,
            "sequence_output": sequence_output,
            0: pooled_output,
            1: sequence_output
        }

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        self.init_pretraining_params(
            exe, self.params_path, main_program=startup_program)

        self.params_layer = {}
        for param in module_program.global_block().iter_parameters():
            param.trainable = trainable
            match = re.match(r'.*layer_(\d+).*', param.name)
            if match:
                # layer num begins from 0
                layer = match.group(1)
                self.params_layer[param.name] = int(layer)

        return inputs, outputs, module_program

    def get_embedding(self, texts, use_gpu=False, batch_size=1):
        """
        get pooled_output and sequence_output for input texts.
        Warnings: this method depends on Paddle Inference Library, it may not work properly in PaddlePaddle < 1.6.2.

        Args:
            texts (list): each element is a text sample, each sample include text_a and text_b where text_b can be omitted.
                          for example: [[sample0_text_a, sample0_text_b], [sample1_text_a, sample1_text_b], ...]
            use_gpu (bool): use gpu or not, default False.
            batch_size (int): the data batch size, default 1.

        Returns:
            pooled_outputs(list): its element is a numpy array, the first feature of each text sample.
            sequence_outputs(list): its element is a numpy array, the whole features of each text sample.
        """
        if not hasattr(self, "emb_job"):
            inputs, outputs, program = self.context(
                trainable=True, max_seq_len=self.MAX_SEQ_LEN)

            reader = hub.reader.ClassifyReader(
                dataset=None,
                vocab_path=self.get_vocab_path(),
                max_seq_len=self.MAX_SEQ_LEN,
                sp_model_path=self.get_spm_path() if hasattr(
                    self, "get_spm_path") else None,
                word_dict_path=self.get_word_dict_path() if hasattr(
                    self, "word_dict_path") else None)

            feed_list = [
                inputs["input_ids"].name,
                inputs["position_ids"].name,
                inputs["segment_ids"].name,
                inputs["input_mask"].name,
            ]

            pooled_feature, seq_feature = outputs["pooled_output"], outputs[
                "sequence_output"]

        if not hasattr(
                self, "emb_job"
        ) or self.emb_job["batch_size"] != batch_size or self.emb_job[
                "use_gpu"] != use_gpu:
            config = hub.RunConfig(
                use_data_parallel=False,
                use_cuda=use_gpu,
                batch_size=batch_size)
            if not hasattr(self, "emb_task"):
                self.emb_job = {}
            self.emb_job["task"] = _BERTEmbeddingTask(
                pooled_feature=pooled_feature,
                seq_feature=seq_feature,
                feed_list=feed_list,
                data_reader=reader,
                config=config,
            )
            self.emb_job["batch_size"] = batch_size
            self.emb_job["use_gpu"] = use_gpu

        return self.emb_job["task"].predict(
            data=texts, return_result=True, accelerate_mode=True)

    def get_vocab_path(self):
        return self.vocab_path

    def get_spm_path(self):
        if hasattr(self, "spm_path"):
            return self.spm_path
        else:
            return None

    def get_word_dict_path(self):
        if hasattr(self, "word_dict_path"):
            return self.word_dict_path
        else:
            return None

    def get_params_layer(self):
        return self.params_layer
