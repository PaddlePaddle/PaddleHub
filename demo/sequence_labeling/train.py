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

if __name__ == '__main__':
    label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
    label_map = {
        idx: label for idx, label in enumerate(label_list)
    }
    model = hub.Module(
        name='ernie_tiny',
        version='2.0.0',
        task='token-cls',
        label_map=label_map,
    )

    train_dataset = hub.datasets.MSRA_NER(
        tokenizer=model.get_tokenizer(),
        max_seq_len=50,
        mode='train'
    )

    dev_dataset = hub.datasets.MSRA_NER(
        tokenizer=model.get_tokenizer(),
        max_seq_len=50,
        mode='dev'
    )

    optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir='token_cls_save_dir', use_gpu=False)

    trainer.train(train_dataset, epochs=1, batch_size=16, eval_dataset=dev_dataset, save_interval=1)
