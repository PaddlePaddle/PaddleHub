#   Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
from paddlehub.datasets.base_nlp_dataset import TextMatchingDataset
from paddlehub.text.bert_tokenizer import BertTokenizer
from paddlehub.text.tokenizer import CustomTokenizer


@download_data(url="https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz")
class LCQMC(TextMatchingDataset):
    label_list = ['0', '1']

    def __init__(
            self,
            tokenizer: Union[BertTokenizer, CustomTokenizer],
            max_seq_len: int = 128,
            mode: str = 'train',
    ):
        base_path = os.path.join(DATA_HOME, "lcqmc")

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


if __name__ == "__main__":
    import paddlehub as hub
    model = hub.Module(name='ernie_tiny')
    tokenizer = model.get_tokenizer()

    ds = LCQMC(tokenizer=tokenizer, max_seq_len=128, mode='dev')
