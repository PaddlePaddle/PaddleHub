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
import os

from paddle.dataset.common import DATA_HOME
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlehub import BertTokenizer
from paddlehub.module.module import moduleinfo, serving
from paddlehub.module.nlp_module import PretrainedModel, register_base_model
from paddlehub.utils.log import logger
from paddlehub.utils.utils import download


class BertEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            seq_length = input_ids.shape[1]
            position_ids = paddle.arange(0, seq_length, dtype="int64")
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Layer):
    """
    """

    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "bert-base-uncased": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "bert-large-uncased": {
            "vocab_size": 30522,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "bert-base-multilingual-uncased": {
            "vocab_size": 105879,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "bert-base-cased": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "bert-base-chinese": {
            "vocab_size": 21128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "bert-base-multilingual-cased": {
            "vocab_size": 119547,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "bert-large-cased": {
            "vocab_size": 28996,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "bert-base-uncased": "https://paddlenlp.bj.bcebos.com/models/transformers/bert-base-uncased.pdparams",
            "bert-large-uncased": "https://paddlenlp.bj.bcebos.com/models/transformers/bert-large-uncased.pdparams",
            "bert-base-multilingual-uncased":
            "http://paddlenlp.bj.bcebos.com/models/transformers/bert-base-multilingual-uncased.pdparams",
            "bert-base-cased": "http://paddlenlp.bj.bcebos.com/models/transformers/bert/bert-base-cased.pdparams",
            "bert-base-chinese": "http://paddlenlp.bj.bcebos.com/models/transformers/bert/bert-base-chinese.pdparams",
            "bert-base-multilingual-cased":
            "http://paddlenlp.bj.bcebos.com/models/transformers/bert/bert-base-multilingual-cased.pdparams",
            "bert-large-cased": "http://paddlenlp.bj.bcebos.com/models/transformers/bert/bert-large-cased.pdparams"
        }
    }
    base_model_prefix = "bert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.bert.config["initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class BertModel(BertPretrainedModel):
    """
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(BertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, hidden_dropout_prob, max_position_embeddings,
                                         type_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e9, axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class BertForSequenceClassification(BertPretrainedModel):
    """
    Model for sentence (pair) classification task with BERT.
    Args:
        bert (BertModel): An instance of BertModel.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Default None
    """

    def __init__(self, bert, num_classes=2, dropout=None):
        super(BertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


@moduleinfo(
    name="bert_multi_uncased_L-12_H-768_A-12",
    version="2.0.0",
    summary=
    "bert_multi_uncased_L-12_H-768_A-12, 12-layer, 768-hidden, 12-heads, 110M parameters. The module is executed as paddle.dygraph.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    type="nlp/semantic_model")
class Bert(nn.Layer):
    """
    BERT model
    """

    def __init__(
            self,
            task=None,
            load_checkpoint=None,
            label_map=None,
    ):
        super(Bert, self).__init__()
        # TODO(zhangxuefei): add token_classification task
        if task == 'sequence_classification':
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path='bert-base-multilingual-uncased')
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy(name='acc_accumulation')
        elif task is None:
            self.model = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-multilingual-uncased')
        else:
            raise RuntimeError("Unknown task %s, task should be sequence_classification" % task)

        self.task = task
        self.label_map = label_map

        if load_checkpoint is not None and os.path.isfile(load_checkpoint):
            state_dict = paddle.load(load_checkpoint)
            self.set_state_dict(state_dict)
            logger.info('Loaded parameters from %s' % os.path.abspath(load_checkpoint))

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, labels=None):
        result = self.model(input_ids, token_type_ids, position_ids, attention_mask)
        if self.task is not None:
            logits = result
            probs = F.softmax(logits, axis=1)
            if labels is not None:
                loss = self.criterion(logits, labels)
                correct = self.metric.compute(probs, labels)
                acc = self.metric.update(correct)
                return probs, loss, acc
            return probs
        else:
            sequence_output, pooled_output = result
            return sequence_output, pooled_output

    def get_vocab_path(self):
        """
        Gets the path of the module vocabulary path.
        """
        save_path = os.path.join(DATA_HOME, 'bert-base-multilingual-uncased',
                                 'bert-base-multilingual-uncased-vocab.txt')
        if not os.path.exists(save_path) or not os.path.isfile(save_path):
            url = "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-multilingual-uncased-vocab.txt"
            download(url, os.path.join(DATA_HOME, 'bert-base-multilingual-uncased'))
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

    def training_step(self, batch: List[paddle.Tensor], batch_idx: int):
        """
        One step for training, which should be called as forward computation.
        Args:
            batch(:obj:List[paddle.Tensor]): The one batch data, which contains the model needed,
                such as input_ids, sent_ids, pos_ids, input_mask and labels.
            batch_idx(int): The index of batch.
        Returns:
            results(:obj: Dict) : The model outputs, such as loss and metrics.
        """
        predictions, avg_loss, acc = self(input_ids=batch[0], token_type_ids=batch[1], labels=batch[2])
        return {'loss': avg_loss, 'metrics': {'acc': acc}}

    def validation_step(self, batch: List[paddle.Tensor], batch_idx: int):
        """
        One step for validation, which should be called as forward computation.
        Args:
            batch(:obj:List[paddle.Tensor]): The one batch data, which contains the model needed,
                such as input_ids, sent_ids, pos_ids, input_mask and labels.
            batch_idx(int): The index of batch.
        Returns:
            results(:obj: Dict) : The model outputs, such as metrics.
        """
        predictions, avg_loss, acc = self(input_ids=batch[0], token_type_ids=batch[1], labels=batch[2])
        return {'metrics': {'acc': acc}}

    def predict(self, data, max_seq_len=128, batch_size=1, use_gpu=False):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(str)`): The processed data whose each element is the raw text.
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            use_gpu(obj:`bool`, defaults to `False`): Whether to use gpu to run or not.

        Returns:
            results(obj:`list`): All the predictions labels.
        """
        # TODO(zhangxuefei): add task token_classification task predict.
        if self.task not in ['sequence_classification']:
            raise RuntimeError("The predict method is for sequence_classification task, but got task %s." % self.task)

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')
        tokenizer = self.get_tokenizer()

        examples = []
        for text in data:
            encoded_inputs = tokenizer.encode(text, max_seq_len=max_seq_len)
            examples.append((encoded_inputs['input_ids'], encoded_inputs['segment_ids']))

        def _batchify_fn(batch):
            input_ids = [entry[0] for entry in batch]
            segment_ids = [entry[1] for entry in batch]
            return input_ids, segment_ids

        # Seperates data into some batches.
        batches = []
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                batches.append(one_batch)
                one_batch = []
        if one_batch:
            # The last batch whose size is less than the config batch_size setting.
            batches.append(one_batch)

        results = []
        self.eval()
        for batch in batches:
            input_ids, segment_ids = _batchify_fn(batch)
            input_ids = paddle.to_tensor(input_ids)
            segment_ids = paddle.to_tensor(segment_ids)

            # TODO(zhangxuefei): add task token_classification postprocess after prediction.
            if self.task == 'sequence_classification':
                probs = self(input_ids, segment_ids)
                idx = paddle.argmax(probs, axis=1).numpy()
                idx = idx.tolist()
                labels = [self.label_map[i] for i in idx]
                results.extend(labels)

        return results

    @serving
    def get_embedding(self, texts, use_gpu=False):
        if self.task is not None:
            raise RuntimeError("The get_embedding method is only valid when task is None, but got task %s" % self.task)

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        tokenizer = self.get_tokenizer()
        results = []
        for text in texts:
            encoded_inputs = tokenizer.encode(text, pad_to_max_seq_len=False)
            input_ids = paddle.to_tensor(encoded_inputs['input_ids']).unsqueeze(0)
            segment_ids = paddle.to_tensor(encoded_inputs['segment_ids']).unsqueeze(0)
            sequence_output, pooled_output = self(input_ids, segment_ids)

            sequence_output = sequence_output.squeeze(0)
            pooled_output = pooled_output.squeeze(0)
            results.append((sequence_output.numpy().tolist(), pooled_output.numpy().tolist()))
        return results


if __name__ == "__main__":
    import numpy as np
    import paddlehub as hub
    src_ids = paddle.to_tensor(np.array([[1, 2, 3, 4, 5]], dtype=np.int64))
    sent_ids = paddle.to_tensor(np.array([[0, 0, 0, 0, 0]], dtype=np.int64))
    paddle.set_device('cpu')

    bert = hub.Module(
        directory=
        '/mnt/zhangxuefei/program-paddle/PaddleHub/modules/text/language_model/bert_multi_uncased_L_12_H_768_A_12',
        version='2.0.0')
    sequence_output, pooled_output = bert(src_ids, sent_ids)  #, pos_ids, input_mask)
    print(sequence_output.shape)
    print(pooled_output.shape)
    vocab_path = bert.get_vocab_path()
