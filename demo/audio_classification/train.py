# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import ast

import paddle

import paddlehub as hub
from paddlehub.datasets import ESC50

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=50, help="Number of epoches for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="Directory to model checkpoint")
parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every n epoch.")
args = parser.parse_args()

if __name__ == "__main__":
    model = hub.Module(name='panns_cnn14', task='sound-cls', num_class=ESC50.num_class)

    train_dataset = ESC50(mode='train')
    dev_dataset = ESC50(mode='dev')

    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters())

    trainer = hub.Trainer(model, optimizer, checkpoint_dir=args.checkpoint_dir, use_gpu=args.use_gpu)
    trainer.train(
        train_dataset,
        epochs=args.num_epoch,
        batch_size=args.batch_size,
        eval_dataset=dev_dataset,
        save_interval=args.save_interval,
    )
