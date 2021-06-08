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

import paddle
import paddlehub as hub
from paddlehub.datasets import ChnSentiCorp
from paddlenlp.data import JiebaTokenizer
from model import BoWModel

import ast
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--hub_embedding_name", type=str, default='w2v_baidu_encyclopedia_target_word-word_dim300', help="")
parser.add_argument("--num_epoch", type=int, default=10, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate used to train with warmup.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="Directory to model checkpoint")
parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every n epoch.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False")

args = parser.parse_args()

if __name__ == '__main__':
    embedder = hub.Module(name=args.hub_embedding_name)
    tokenizer = embedder.get_tokenizer()

    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='dev')
    test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=args.max_seq_len, mode='test')

    model = BoWModel(embedder=embedder)
    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir=args.checkpoint_dir, use_gpu=args.use_gpu)
    trainer.train(
        train_dataset,
        epochs=args.num_epoch,
        batch_size=args.batch_size,
        eval_dataset=dev_dataset,
        save_interval=args.save_interval,
    )
    trainer.evaluate(test_dataset, batch_size=args.batch_size)
