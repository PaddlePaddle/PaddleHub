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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from paddlehub import TransformerModule
from paddlehub.module.module import moduleinfo

from chinese_roberta_wwm_ext.model.bert import BertConfig, BertModel


@moduleinfo(
    name="chinese-roberta-wwm-ext",
    version="1.0.0",
    summary=
    "chinese-roberta-wwm-ext, 12-layer, 768-hidden, 12-heads, 110M parameters ",
    author="ymcui",
    author_email="ymcui@ir.hit.edu.cn",
    type="nlp/semantic_model",
)
class BertWwm(TransformerModule):
    def _initialize(self):
        self.MAX_SEQ_LEN = 512
        self.params_path = os.path.join(self.directory, "assets", "params")
        self.vocab_path = os.path.join(self.directory, "assets", "vocab.txt")

        bert_config_path = os.path.join(self.directory, "assets",
                                        "bert_config.json")
        self.bert_config = BertConfig(bert_config_path)

    def net(self, input_ids, position_ids, segment_ids, input_mask):
        """
        create neural network.

        Args:
            input_ids (tensor): the word ids.
            position_ids (tensor): the position ids.
            segment_ids (tensor): the segment ids.
            input_mask (tensor): the padding mask.

        Returns:
            pooled_output (tensor):  sentence-level output for classification task.
            sequence_output (tensor): token-level output for sequence task.
        """
        bert = BertModel(
            src_ids=input_ids,
            position_ids=position_ids,
            sentence_ids=segment_ids,
            input_mask=input_mask,
            config=self.bert_config,
            use_fp16=False)
        pooled_output = bert.get_pooled_output()
        sequence_output = bert.get_sequence_output()
        return pooled_output, sequence_output


if __name__ == '__main__':
    test_module = BertWwm()
