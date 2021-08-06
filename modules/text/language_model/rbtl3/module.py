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

from paddlenlp.transformers.roberta.modeling import RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer
from paddlenlp.metrics import ChunkEvaluator
from paddlehub.module.module import moduleinfo
from paddlehub.module.nlp_module import TransformerModule
from paddlehub.utils.log import logger


@moduleinfo(
    name="rbtl3",
    version="2.0.1",
    summary="rbtl3, 3-layer, 1024-hidden, 16-heads, 61M parameters ",
    author="ymcui",
    author_email="ymcui@ir.hit.edu.cn",
    type="nlp/semantic_model",
    meta=TransformerModule,
)
class Roberta(nn.Layer):
    """
    RoBERTa model
    """

    def __init__(
            self,
            task: str = None,
            load_checkpoint: str = None,
            label_map: Dict = None,
            num_classes: int = 2,
            suffix: bool = False,
            **kwargs,
    ):
        super(Roberta, self).__init__()
        if label_map:
            self.label_map = label_map
            self.num_classes = len(label_map)
        else:
            self.num_classes = num_classes

        if task == 'sequence_classification':
            task = 'seq-cls'
            logger.warning(
                "current task name 'sequence_classification' was renamed to 'seq-cls', "
                "'sequence_classification' has been deprecated and will be removed in the future.", )
        if task == 'seq-cls':
            self.model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path='rbtl3', num_classes=self.num_classes, **kwargs)
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy()
        elif task == 'token-cls':
            self.model = RobertaForTokenClassification.from_pretrained(
                pretrained_model_name_or_path='rbtl3', num_classes=self.num_classes, **kwargs)
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = ChunkEvaluator(label_list=[self.label_map[i] for i in sorted(self.label_map.keys())], suffix=suffix)
        elif task == 'text-matching':
            self.model = RobertaModel.from_pretrained(pretrained_model_name_or_path='rbtl3', **kwargs)
            self.dropout = paddle.nn.Dropout(0.1)
            self.classifier = paddle.nn.Linear(self.model.config['hidden_size'] * 3, 2)
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy()
        elif task is None:
            self.model = RobertaModel.from_pretrained(pretrained_model_name_or_path='rbtl3', **kwargs)
        else:
            raise RuntimeError("Unknown task {}, task should be one in {}".format(task, self._tasks_supported))

        self.task = task

        if load_checkpoint is not None and os.path.isfile(load_checkpoint):
            state_dict = paddle.load(load_checkpoint)
            self.set_state_dict(state_dict)
            logger.info('Loaded parameters from %s' % os.path.abspath(load_checkpoint))

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                query_input_ids=None,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_input_ids=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None,
                seq_lengths=None,
                labels=None):

        if self.task != 'text-matching':
            result = self.model(input_ids, token_type_ids, position_ids, attention_mask)
        else:
            query_result = self.model(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)
            title_result = self.model(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)

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
                self.metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
                _, _, f1_score = map(float, self.metric.accumulate())
                return token_level_probs, loss, {'f1_score': f1_score}
            return token_level_probs
        elif self.task == 'text-matching':
            query_token_embedding, _ = query_result
            query_token_embedding = self.dropout(query_token_embedding)
            query_attention_mask = paddle.unsqueeze(
                (query_input_ids != self.model.pad_token_id).astype(self.model.pooler.dense.weight.dtype), axis=2)
            query_token_embedding = query_token_embedding * query_attention_mask
            query_sum_embedding = paddle.sum(query_token_embedding, axis=1)
            query_sum_mask = paddle.sum(query_attention_mask, axis=1)
            query_mean = query_sum_embedding / query_sum_mask

            title_token_embedding, _ = title_result
            title_token_embedding = self.dropout(title_token_embedding)
            title_attention_mask = paddle.unsqueeze(
                (title_input_ids != self.model.pad_token_id).astype(self.model.pooler.dense.weight.dtype), axis=2)
            title_token_embedding = title_token_embedding * title_attention_mask
            title_sum_embedding = paddle.sum(title_token_embedding, axis=1)
            title_sum_mask = paddle.sum(title_attention_mask, axis=1)
            title_mean = title_sum_embedding / title_sum_mask

            sub = paddle.abs(paddle.subtract(query_mean, title_mean))
            projection = paddle.concat([query_mean, title_mean, sub], axis=-1)
            logits = self.classifier(projection)
            probs = F.softmax(logits)
            if labels is not None:
                loss = self.criterion(logits, labels)
                correct = self.metric.compute(probs, labels)
                acc = self.metric.update(correct)
                return probs, loss, {'acc': acc}
            return probs
        else:
            sequence_output, pooled_output = result
            return sequence_output, pooled_output

    @staticmethod
    def get_tokenizer(*args, **kwargs):
        """
        Gets the tokenizer that is customized for this module.
        """
        return RobertaTokenizer.from_pretrained(pretrained_model_name_or_path='rbtl3', *args, **kwargs)
