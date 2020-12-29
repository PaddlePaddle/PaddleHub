# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict
import os
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers.bert.modeling import BertForSequenceClassification, BertModel, BertForTokenClassification
from paddlenlp.transformers.bert.tokenizer import BertTokenizer
from paddlenlp.metrics import ChunkEvaluator
from paddlehub.datasets.base_nlp_dataset import ChunkScheme
from paddlehub.module.module import moduleinfo
from paddlehub.module.nlp_module import TransformerModule
from paddlehub.utils.log import logger


@moduleinfo(
    name="bert-large-cased",
    version="2.0.1",
    summary=
    "bert_cased_L-24_H-1024_A-16, 24-layer, 1024-hidden, 16-heads, 340M parameters. The module is executed as paddle.dygraph.",
    author="paddlepaddle",
    author_email="",
    type="nlp/semantic_model",
    meta=TransformerModule)
class Bert(nn.Layer):
    """
    BERT model
    """

    def __init__(
            self,
            task: str = None,
            load_checkpoint: str = None,
            label_map: Dict = None,
            num_classes: int = 2,
            chunk_scheme: ChunkScheme = ChunkScheme.IOB,
            **kwargs,
    ):
        super(Bert, self).__init__()
        if label_map:
            self.label_map = label_map
            self.num_classes = len(label_map)
        else:
            self.num_classes = num_classes

        if task == 'sequence_classification':
            task = 'seq-cls'
            logger.warning(
                "current task name 'sequence_classification' was renamed to 'seq-cls', "
                "'sequence_classification' has been deprecated and will be removed in the future.",
            )
        if task == 'seq-cls':
            self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-large-cased', num_classes=self.num_classes, **kwargs)
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy()
        elif task == 'token-cls':
            self.model = BertForTokenClassification.from_pretrained(pretrained_model_name_or_path='bert-large-cased', num_classes=self.num_classes, **kwargs)
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = ChunkEvaluator(
                num_chunk_types=int(math.ceil((self.num_classes-1)/2.0)),
                chunk_scheme=chunk_scheme.value,
            )
        elif task is None:
            self.model = BertModel.from_pretrained(pretrained_model_name_or_path='bert-large-cased', **kwargs)
        else:
            raise RuntimeError("Unknown task {}, task should be one in {}".format(
                task, self._tasks_supported))

        self.task = task

        if load_checkpoint is not None and os.path.isfile(load_checkpoint):
            state_dict = paddle.load(load_checkpoint)
            self.set_state_dict(state_dict)
            logger.info('Loaded parameters from %s' % os.path.abspath(load_checkpoint))

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, seq_lengths=None, labels=None):
        result = self.model(input_ids, token_type_ids, position_ids, attention_mask)
        if self.task == 'seq-cls':
            logits = result
            probs = F.softmax(logits, axis=1)
            if labels is not None:
                loss = self.criterion(logits, labels)
                correct = self.metric.compute(probs, labels)
                acc = self.metric.update(correct)
                return probs, loss, {'acc': acc}
            return probs
        elif self.task == 'token-cls':
            logits = result
            token_level_probs = F.softmax(logits, axis=-1)
            preds = token_level_probs.argmax(axis=-1)
            if labels is not None:
                loss = self.criterion(logits, labels.unsqueeze(-1))
                num_infer_chunks, num_label_chunks, num_correct_chunks = \
                    self.metric.compute(None, seq_lengths, preds, labels)
                self.metric.update(
                    num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
                _, _, f1_score = map(float, self.metric.accumulate())
                return token_level_probs, loss, {'f1_score': f1_score}
            return token_level_probs
        else:
            sequence_output, pooled_output = result
            return sequence_output, pooled_output

    @staticmethod
    def get_tokenizer(*args, **kwargs):
        """
        Gets the tokenizer that is customized for this module.
        """
        return BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-large-cased', *args, **kwargs)
