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

from paddlehub import ErnieTinyTokenizer
from paddlehub.module.module import moduleinfo, serving
from paddlehub.module.nlp_module import PretrainedModel, register_base_model
from paddlehub.utils.log import logger
from paddlehub.utils.utils import download


class ErnieEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 pad_token_id=0):
        super(ErnieEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
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


class ErniePooler(nn.Layer):
    """
    """

    def __init__(self, hidden_size):
        super(ErniePooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErniePretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ERNIE models. It provides ERNIE related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18000,
            "pad_token_id": 0,
        },
        "ernie_tiny": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 600,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 50006,
            "pad_token_id": 0,
        },
        "ernie_v2_eng_base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie_v2_eng_large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams",
            "ernie_tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/ernie_tiny.pdparams",
            "ernie_v2_eng_base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_base/ernie_v2_eng_base.pdparams",
            "ernie_v2_eng_large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_large/ernie_v2_eng_large.pdparams",
        }
    }
    base_model_prefix = "ernie"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.ernie.config["initializer_range"],
                    shape=layer.weight.shape))


@register_base_model
class ErnieModel(ErniePretrainedModel):
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
        super(ErnieModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ErnieEmbeddings(vocab_size, hidden_size, hidden_dropout_prob, max_position_embeddings,
                                          type_vocab_size, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErniePooler(hidden_size)
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


class ErnieForSequenceClassification(ErniePretrainedModel):
    """
    Model for sentence (pair) classification task with ERNIE.
    Args:
        ernie (ErnieModel): An instance of `ErnieModel`.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of ERNIE.
            If None, use the same value as `hidden_dropout_prob` of `ErnieModel`
            instance `Ernie`. Default None
    """

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, pooled_output = self.ernie(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


@moduleinfo(
    name="ernie_tiny",
    version="2.0.0",
    summary="Baidu's ERNIE-tiny, Enhanced Representation through kNowledge IntEgration, tiny version, max_seq_len=512",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    type="nlp/semantic_model")
class ErnieTiny(nn.Layer):
    """
    Ernie model
    """

    def __init__(
            self,
            task=None,
            load_checkpoint=None,
            label_map=None,
    ):
        super(ErnieTiny, self).__init__()
        # TODO(zhangxuefei): add token_classification task
        if task == 'sequence_classification':
            self.model = ErnieForSequenceClassification.from_pretrained(pretrained_model_name_or_path='ernie_tiny')
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            self.metric = paddle.metric.Accuracy(name='acc_accumulation')
        elif task is None:
            self.model = ErnieModel.from_pretrained(pretrained_model_name_or_path='ernie_tiny')
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
        save_path = os.path.join(DATA_HOME, 'ernie_tiny', 'vocab.txt')
        if not os.path.exists(save_path) or not os.path.isfile(save_path):
            url = "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/vocab.txt"
            download(url, os.path.join(DATA_HOME, 'ernie_tiny'))
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
        spm_path = os.path.join(DATA_HOME, 'ernie_tiny', 'spm_cased_simp_sampled.model')
        if not os.path.exists(spm_path) or not os.path.isfile(spm_path):
            url = "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/spm_cased_simp_sampled.model"
            download(url, os.path.join(DATA_HOME, 'ernie_tiny'))

        word_dict_path = os.path.join(DATA_HOME, 'ernie_tiny', 'dict.wordseg.pickle')
        if not os.path.exists(word_dict_path) or not os.path.isfile(word_dict_path):
            url = "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/dict.wordseg.pickle"
            download(url, os.path.join(DATA_HOME, 'ernie_tiny'))

        return ErnieTinyTokenizer(self.get_vocab_path(), spm_path, word_dict_path)

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

    ernie = hub.Module(
        directory='/mnt/zhangxuefei/program-paddle/PaddleHub/modules/text/language_model/ernie_tiny', version='2.0.0')
    sequence_output, pooled_output = ernie(src_ids, sent_ids)  #, pos_ids, input_mask)
    print(sequence_output.shape)
    print(pooled_output.shape)
    vocab_path = ernie.get_vocab_path()
