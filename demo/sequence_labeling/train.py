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

import paddle
import paddlehub as hub
from paddlehub.datasets import MSRA_NER, MSRA_NER_CHUNK_SCHEME, MSRA_NER_LABEL_LIST

if __name__ == '__main__':
    model = hub.Module(
        name='ernie_tiny',
        version='2.0.1',
        task='token-cls',
        ignore_label=-100,
        chunk_scheme=MSRA_NER_CHUNK_SCHEME,
        num_classes=len(MSRA_NER_LABEL_LIST)
    )

    train_dataset = MSRA_NER(
        tokenizer=model.get_tokenizer(),
        max_seq_len=128,
        ignore_label=-100,
        mode='train'
    )
    dev_dataset = MSRA_NER(
        tokenizer=model.get_tokenizer(),
        max_seq_len=128,
        ignore_label=-100,
        mode='dev'
    )
    test_dataset = MSRA_NER(
        tokenizer=model.get_tokenizer(),
        max_seq_len=128,
        ignore_label=-100,
        mode='test'
    )

    optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir='token_cls_save_dir', use_gpu=True)
    trainer.train(train_dataset, epochs=1, batch_size=8, eval_dataset=dev_dataset, save_interval=1)
    trainer.evaluate(test_dataset, batch_size=32)
