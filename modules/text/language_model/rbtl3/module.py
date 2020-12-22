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
import os

from paddle.dataset.common import DATA_HOME
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlehub import BertTokenizer
from paddlenlp.transformers.roberta.modeling import RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel
from paddlehub.module.module import moduleinfo
from paddlehub.module.nlp_module import TransformerModule
from paddlehub.utils.log import logger
from paddlehub.utils.utils import download


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
            task=None,
            load_checkpoint=None,
            label_map=None,
            num_classes=2,
            **kwargs,
    ):
        super(Roberta, self).__init__()
        if label_map:
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
            self.model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path='rbtl3',
                num_classes=self.num_classes,
                **kwargs
            )
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy()
        elif task == 'token-cls':
            self.model = RobertaForTokenClassification.from_pretrained(
                pretrained_model_name_or_path='rbtl3',
                num_classes=self.num_classes,
                **kwargs
            )
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy()
        elif task is None:
            self.model = RobertaModel.from_pretrained(pretrained_model_name_or_path='rbtl3', **kwargs)
        else:
            raise RuntimeError("Unknown task {}, task should be one in {}".format(
                task, self._tasks_supported))

        self.task = task
        self.label_map = label_map

        if load_checkpoint is not None and os.path.isfile(load_checkpoint):
            state_dict = paddle.load(load_checkpoint)
            self.set_state_dict(state_dict)
            logger.info('Loaded parameters from %s' % os.path.abspath(load_checkpoint))

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, labels=None):
        result = self.model(input_ids, token_type_ids, position_ids, attention_mask)
        if self.task == 'seq-cls':
            logits = result
            probs = F.softmax(logits, axis=1)
            if labels is not None:
                loss = self.criterion(logits, labels)
                correct = self.metric.compute(probs, labels)
                acc = self.metric.update(correct)
                return probs, loss, acc
            return probs
        elif self.task == 'token-cls':
            logits = result
            token_level_probs = F.softmax(logits, axis=2)
            if labels is not None:
                labels = paddle.to_tensor(labels).unsqueeze(-1)
                loss = self.criterion(logits, labels)
                correct = self.metric.compute(token_level_probs, labels)
                acc = self.metric.update(correct)
                return token_level_probs, loss, acc
            return token_level_probs
        else:
            sequence_output, pooled_output = result
            return sequence_output, pooled_output

    def get_vocab_path(self):
        """
        Gets the path of the module vocabulary path.
        """
        save_path = os.path.join(DATA_HOME, 'rbtl3', 'vocab.txt')
        if not os.path.exists(save_path) or not os.path.isfile(save_path):
            url = "https://paddlenlp.bj.bcebos.com/models/transformers/rbtl3/vocab.txt"
            download(url, os.path.join(DATA_HOME, 'rbtl3'))
        return save_path

    def get_tokenizer(self, tokenize_chinese_chars=True):
        """
        Gets the tokenizer that is customized for this module.
        Args:
            tokenize_chinese_chars (:obj: bool , defaults to :obj: True):
                Whether to tokenize chinese characters or not.
        Returns:
            tokenizer (:obj:BertTokenizer) : The tokenizer which was customized for this module.
        """
        return BertTokenizer(tokenize_chinese_chars=tokenize_chinese_chars, vocab_file=self.get_vocab_path())
