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
from paddlehub.module.modeling_ernie import ErnieModel, ErnieForSequenceClassification
from paddlehub.module.module import moduleinfo, serving
from paddlehub.utils.log import logger
from paddlehub.utils.utils import download


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
