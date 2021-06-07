#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from typing import Union
import os

from paddlehub.env import DATA_HOME
from paddlehub.utils.download import download_data
from paddlehub.datasets.base_nlp_dataset import SeqLabelingDataset
from paddlehub.text.bert_tokenizer import BertTokenizer
from paddlehub.text.tokenizer import CustomTokenizer


@download_data(url="https://bj.bcebos.com/paddlehub-dataset/msra_ner.tar.gz")
class MSRA_NER(SeqLabelingDataset):
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """
    label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    def __init__(
            self,
            tokenizer: Union[BertTokenizer, CustomTokenizer],
            max_seq_len: int = 128,
            mode: str = 'train',
    ):
        base_path = os.path.join(DATA_HOME, "msra_ner")

        if mode == 'train':
            data_file = 'train.tsv'
        elif mode == 'test':
            data_file = 'test.tsv'
        else:
            data_file = 'dev.tsv'
        super().__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_file=None,
            label_list=self.label_list,
            is_file_with_header=True,
        )
