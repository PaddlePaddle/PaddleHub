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

from typing import Generic, List

import paddle
import numpy as np

from paddlehub.compat.task.config import RunConfig
from paddlehub.compat.task.base_task import BaseTask
from paddlehub.compat.task.task_utils import RunState


class TransformerEmbeddingTask(BaseTask):
    def __init__(self,
                 pooled_feature: paddle.static.Variable,
                 seq_feature: paddle.static.Variable,
                 feed_list: List[str],
                 data_reader: Generic,
                 config: RunConfig = None):
        main_program = pooled_feature.block.program
        super(TransformerEmbeddingTask, self).__init__(
            main_program=main_program, config=config, feed_list=feed_list, data_reader=data_reader, metrics_choices=[])
        self.pooled_feature = pooled_feature
        self.seq_feature = seq_feature

    def _build_net(self) -> List[paddle.static.Variable]:
        # ClassifyReader will return the seqence length of an input text
        self.seq_len = paddle.static.data(name='seq_len', shape=[1], dtype='int64', lod_level=0)
        return [self.pooled_feature, self.seq_feature]

    def _postprocessing(self, run_states: List[RunState]) -> List[List[np.ndarray]]:
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results
            batch_pooled_features = batch_result[0]
            batch_seq_features = batch_result[1]
            for i in range(len(batch_pooled_features)):
                results.append([batch_pooled_features[i], batch_seq_features[i]])
        return results

    @property
    def feed_list(self) -> List[str]:
        feed_list = [varname for varname in self._base_feed_list] + [self.seq_len.name]
        return feed_list

    @property
    def fetch_list(self) -> List[str]:
        fetch_list = [output.name for output in self.outputs] + [self.seq_len.name]
        return fetch_list
