# coding:utf-8
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
from typing import Dict, List, Optional, Union, Tuple
import os

from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
from paddlehub.tokenizer.tokenizer import CustomTokenizer

from .base_nlp_dataset import TextClassificationDataset
from paddlehub.env import DATA_HOME

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz"


class ChnSentiCorp(TextClassificationDataset):
    """
    ChnSentiCorp is a dataset for chinese sentiment classification,
    which was published by Tan Songbo at ICT of Chinese Academy of Sciences.
    """

    def __init__(self, tokenizer: Union[BertTokenizer, CustomTokenizer], max_seq_len: int = 128, mode: str = 'train'):
        """
        Args:
            tokenizer (:obj:`BertTokenizer` or :obj:`CustomTokenizer`):
                It tokenizes the text and encodes the data as model needed.
            max_seq_len (:obj:`int`, `optional`, defaults to :128):
                The maximum length (in number of tokens) for the inputs to the selected module,
                such as ernie, bert and so on.
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train, test or dev).

        Examples:
            .. code-block:: python
                from paddlehub.datasets.chnsenticorp import ChnSentiCorp
                from paddlehub.tokenizer.bert_tokenizer import BertTokenizer

                tokenizer = BertTokenizer(vocab_file='./vocab.txt')

                train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=120, mode='train')
                dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=120, mode='dev')
                test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=120, mode='test')
        """
        base_path = os.path.join(DATA_HOME, "chnsenticorp")
        self._download_and_uncompress_dataset(base_path, url=_DATA_URL)
        if mode == 'train':
            data_file = 'train.tsv'
        elif mode == 'test':
            data_file = 'test.tsv'
        else:
            data_file = 'dev.tsv'
        super(ChnSentiCorp, self).__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=["0", "1"],
            is_file_with_header=True)


if __name__ == "__main__":
    tokenizer = BertTokenizer(vocab_file='./vocab.txt')
    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='dev')
    test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='test')

    index = 0
    while index < 3:
        record = train_dataset.__getitem__(index)
        print("train record: ", record)
        record = dev_dataset.__getitem__(index)
        print("dev record: ", record)
        record = test_dataset.__getitem__(index)
        print("test record: ", record)
        index += 1
