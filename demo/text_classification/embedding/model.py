# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as nlp
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.data import JiebaTokenizer

from paddlehub.utils.log import logger
from paddlehub.utils.utils import pad_sequence, trunc_sequence


class BoWModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size (obj:`int`): The vocabulary size.
        emb_dim (obj:`int`, optional, defaults to 300):  The embedding dimension.
        hidden_size (obj:`int`, optional, defaults to 128): The first full-connected layer hidden size.
        fc_hidden_size (obj:`int`, optional, defaults to 96): The second full-connected layer hidden size.
        num_classes (obj:`int`): All the labels that the data has.
    """

    def __init__(self,
                 num_classes: int = 2,
                 embedder: TokenEmbedding = None,
                 tokenizer: JiebaTokenizer = None,
                 hidden_size: int = 128,
                 fc_hidden_size: int = 96,
                 load_checkpoint: str = None,
                 label_map: dict = None):
        super().__init__()
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.label_map = label_map

        emb_dim = self.embedder.embedding_dim
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.dropout = nn.Dropout(p=0.3, axis=1)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)
        self.criterion = nn.loss.CrossEntropyLoss()
        self.metric = paddle.metric.Accuracy()

        if load_checkpoint is not None and os.path.isfile(load_checkpoint):
            state_dict = paddle.load(load_checkpoint)
            self.set_state_dict(state_dict)
            logger.info('Loaded parameters from %s' % os.path.abspath(load_checkpoint))

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
        _, avg_loss, metric = self(ids=batch[0], labels=batch[1])
        self.metric.reset()
        return {'loss': avg_loss, 'metrics': metric}

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
        _, _, metric = self(ids=batch[0], labels=batch[1])
        self.metric.reset()
        return {'metrics': metric}

    def forward(self, ids: paddle.Tensor, labels: paddle.Tensor = None):

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(ids)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)
        summed = self.dropout(summed)
        encoded_text = paddle.tanh(summed)

        # Shape: (batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc2_out)

        probs = F.softmax(logits, axis=1)
        if labels is not None:
            loss = self.criterion(logits, labels)
            correct = self.metric.compute(probs, labels)
            acc = self.metric.update(correct)
            return probs, loss, {'acc': acc}
        else:
            return probs

    def _batchify(self, data: List[List[str]], max_seq_len: int, batch_size: int):
        examples = []
        for item in data:
            ids = self.tokenizer.encode(sentence=item[0])

            if len(ids) > max_seq_len:
                ids = trunc_sequence(ids, max_seq_len)
            else:
                pad_token = self.tokenizer.vocab.pad_token
                pad_token_id = self.tokenizer.vocab.to_indices(pad_token)
                ids = pad_sequence(ids, max_seq_len, pad_token_id)
            examples.append(ids)

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield one_batch
                one_batch = []
        if one_batch:
            # The last batch whose size is less than the config batch_size setting.
            yield one_batch

    def predict(
            self,
            data: List[List[str]],
            max_seq_len: int = 128,
            batch_size: int = 1,
            use_gpu: bool = False,
            return_result: bool = True,
    ):
        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        batches = self._batchify(data, max_seq_len, batch_size)
        results = []
        self.eval()
        for batch in batches:
            ids = paddle.to_tensor(batch)
            probs = self(ids)
            idx = paddle.argmax(probs, axis=1).numpy()

            if return_result:
                idx = idx.tolist()
                labels = [self.label_map[i] for i in idx]
                results.extend(labels)
            else:
                results.extend(probs.numpy())

        return results
