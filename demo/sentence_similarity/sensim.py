#coding:utf-8
#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
"""similarity between two sentences"""

import numpy as np
import scipy
from scipy.spatial import distance

from paddlehub.reader.tokenization import load_vocab, convert_tokens_to_ids
import paddle.fluid as fluid
import paddlehub as hub


def convert_tokens_to_ids(vocab, text):
    wids = []
    tokens = text.split(" ")
    for token in tokens:
        wid = vocab.get(token, None)
        if not wid:
            wid = vocab["unknown"]
        wids.append(wid)
    return wids


if __name__ == "__main__":

    module = hub.Module(name="word2vec_skipgram")
    inputs, outputs, program = module.context(trainable=False)
    vocab = load_vocab(module.get_vocab_path())

    word_ids = inputs["word_ids"]
    embedding = outputs["word_embs"]

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[word_ids], place=place)
    w2v, = exe.run(
        program,
        feed=feeder.feed([[[1123]]]),
        fetch_list=[embedding.name],
        return_numpy=False)

    data = [
        [
            "驾驶 违章 一次 扣 12分 用 两个 驾驶证 处理 可以 吗",
            " 一次性 扣 12分 的 违章 , 能用 不满 十二分 的 驾驶证 扣分 吗"
        ],
        ["水果 放 冰箱 里 储存 好 吗", "中国银行 纪念币 网上 怎么 预约"],
        ["电脑 反应 很 慢 怎么 办", "反应 速度 慢 , 电脑 总是 卡 是 怎么回事"],
    ]

    for item in data:
        text_a = convert_tokens_to_ids(vocab, item[0])
        text_b = convert_tokens_to_ids(vocab, item[1])

        vecs_a, = exe.run(
            program,
            feed=feeder.feed([[text_a]]),
            fetch_list=[embedding.name],
            return_numpy=False)
        vecs_a = np.array(vecs_a)
        vecs_b, = exe.run(
            program,
            feed=feeder.feed([[text_b]]),
            fetch_list=[embedding.name],
            return_numpy=False)
        vecs_b = np.array(vecs_b)

        sent_emb_a = np.sum(vecs_a, axis=0)
        sent_emb_b = np.sum(vecs_b, axis=0)
        cos_sim = 1 - distance.cosine(sent_emb_a, sent_emb_b)

        print("text_a: %s; text_b: %s; cosine_similarity: %.5f" %
              (item[0], item[1], cos_sim))
