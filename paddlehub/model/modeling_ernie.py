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
from typing import Dict, List, Optional, Union, Tuple
import json

import paddle.fluid.dygraph as dygraph
import paddle.fluid as fluid

from paddlehub.utils.log import logger
from paddlehub.module.module import moduleinfo, Module
from paddlehub.tokenizer.bert_tokenizer import BertTokenizer


class ErnieforSequenceClassification(dygraph.Layer):
    """
    This model is customized for that enrie module does the sequence classification, such as the text sentiment classification.
    This model exploits the ernie + full-connect layers (softmax as activation) to classify the texts.
    """

    def __init__(self, ernie_module: Module, num_classes: int):
        """
        Args:
            ernie_module (:obj: Module): The ERNIE module loaded by PaddleHub.
            num_class (:obj:int): The total number of labels of the text classification task.

        Examples:
            .. code-block:: python
                from paddlehub.model.modeling_ernie import ErnieforSequenceClassification
                import paddlehub as hub

                ernie = hub.Module(name='ernie', version='2.0.0')
                tokenizer = ernie.get_tokenizer()
                model = ErnieforSequenceClassification(ernie_module=ernie, num_classes=2)
        """
        dygraph.Layer.__init__(self)

        self.module = ernie_module

        self.num_classes = num_classes

        self.dropout = lambda x: fluid.layers.dropout(
            x, dropout_prob=0.1, dropout_implementation="upscale_in_train") if self.training else x
        self.prediction = dygraph.Linear(
            input_dim=768,
            output_dim=self.num_classes,
            param_attr=fluid.ParamAttr(name="cls_out_w", initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr="cls_out_b",
            act="softmax")

    def forward(self,
                input_ids: fluid.Variable,
                sent_ids: fluid.Variable,
                pos_ids: fluid.Variable,
                input_mask: fluid.Variable,
                labels: fluid.Variable = None):
        """
        Args:
            input_ids (:obj:`Variable` of shape `[batch_size, seq_len]`):
                Indices of input sequence tokens in the vocabulary.
            sent_ids (:obj: `Variable` of shape `[batch_size, seq_len]`):
                Aka token_type_ids, Segment token indices to indicate first and second portions of the inputs.
                if None, assume all tokens come from `segment_a`
            pos_ids(:obj: `Variable` of shape `[batch_size, seq_len]`):
                Indices of positions of each input sequence tokens in the position embeddings.
            input_mask(:obj: `Variable` of shape `[batch_size, seq_len]`):
                Mask to avoid performing attention on the padding token indices of the encoder input.
            labels(:obj: `Variable` of shape `[batch_size, 1]`, `optional`, defaults to :obj`None`):
                Labels for computing the sequence classification/regression loss.

        Returns:
            The predictions of input data.
            If labels as input not None, then it will return the averge loss and accauracy.
        """

        pooled_output, sequence_output = self.module(input_ids, sent_ids, pos_ids, input_mask)
        cls_feats = self.dropout(pooled_output)
        predictions = self.prediction(cls_feats)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = fluid.layers.reshape(labels, [-1, 1])
            loss = fluid.layers.cross_entropy(input=predictions, label=labels)
            avg_loss = fluid.layers.mean(loss)
            acc = fluid.layers.accuracy(input=predictions, label=labels)
            return predictions, avg_loss, acc
        else:
            return predictions

    def training_step(self, batch: List[fluid.Variable], batch_idx: int):
        """
        One step for training, which should be called as forward computation.

        Args:
            batch(:obj:List[fluid.Variable]): The one batch data, which contains the model needed,
                such as input_ids, sent_ids, pos_ids, input_mask and labels.
            batch_idx(int): The index of batch.
        Returns:
            results(:obj: Dict) : The model outputs, such as loss and metrics.

        """
        predictions, avg_loss, acc = self(
            input_ids=batch[0], sent_ids=batch[1], pos_ids=batch[2], input_mask=batch[3], labels=batch[4])
        return {'loss': avg_loss, 'metrics': {'acc': acc}}

    def validation_step(self, batch, batch_idx):
        """
        One step for validation, which should be called as forward computation.

        Args:
            batch(:obj:List[fluid.Variable]): The one batch data, which contains the model needed,
                such as input_ids, sent_ids, pos_ids, input_mask and labels.
            batch_idx(int): The index of batch.

        Returns:
            results(:obj: Dict) : The model outputs, such as metrics.
        """
        predictions, avg_loss, acc = self(
            input_ids=batch[0], sent_ids=batch[1], pos_ids=batch[2], input_mask=batch[3], labels=batch[4])
        return {'metrics': {'acc': acc}}
