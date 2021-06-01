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

import paddlehub as hub
from paddlenlp.data import JiebaTokenizer
from model import BoWModel

import ast
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--hub_embedding_name", type=str, default='w2v_baidu_encyclopedia_target_word-word_dim300', help="")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint", type=str, default='./checkpoint/best_model/model.pdparams', help="Model checkpoint")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning, input should be True or False")

args = parser.parse_args()

if __name__ == '__main__':
    # Data to be prdicted
    data = [
        ["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"],
        ["交通方便；环境很好；服务态度很好 房间较小"],
        ["还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。"],
        ["前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级设施，下次不会再选择入住此店啦"],
        ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"],
    ]

    label_map = {0: 'negative', 1: 'positive'}

    embedder = hub.Module(name=args.hub_embedding_name)
    tokenizer = embedder.get_tokenizer()
    model = BoWModel(embedder=embedder, tokenizer=tokenizer, load_checkpoint=args.checkpoint, label_map=label_map)

    results = model.predict(
        data, max_seq_len=args.max_seq_len, batch_size=args.batch_size, use_gpu=args.use_gpu, return_result=False)
    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {}'.format(text[0], results[idx]))
