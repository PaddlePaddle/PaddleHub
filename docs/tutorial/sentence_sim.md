# 使用Word2Vec进行文本语义相似度计算

本示例展示利用PaddleHub“端到端地”完成文本相似度计算。

## 一、准备文本数据

如
```
驾驶违章一次扣12分用两个驾驶证处理可以吗    一次性扣12分的违章,能用不满十二分的驾驶证扣分吗
水果放冰箱里储存好吗    中国银行纪念币网上怎么预约
电脑反应很慢怎么办    反应速度慢,电脑总是卡是怎么回事
```

## 二、分词
利用PaddleHub Module LAC对文本数据进行分词。

```python
# coding:utf-8
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

from paddlehub.reader.tokenization import load_vocab
import paddle.fluid as fluid
import paddlehub as hub

raw_data = [
    ["驾驶违章一次扣12分用两个驾驶证处理可以吗", "一次性扣12分的违章,能用不满十二分的驾驶证扣分吗"],
    ["水果放冰箱里储存好吗", "中国银行纪念币网上怎么预约"],
    ["电脑反应很慢怎么办", "反应速度慢,电脑总是卡是怎么回事"]
]

lac = hub.Module(name="lac")

processed_data = []
for text_pair in raw_data:
    inputs = {"text" : text_pair}
    results = lac.lexical_analysis(data=inputs, use_gpu=True, batch_size=2)
    data = []
    for result in results:
        data.append(" ".join(result["word"]))
    processed_data.append(data)
```

## 三、计算文本语义相似度

将分词文本中的单词相应替换为wordid，之后输入wor2vec module中计算两个文本语义相似度。

```python
def convert_tokens_to_ids(vocab, text):
    wids = []
    tokens = text.split(" ")
    for token in tokens:
        wid = vocab.get(token, None)
        if not wid:
            wid = vocab["unknown"]
        wids.append(wid)
    return wids

module = hub.Module(name="word2vec_skipgram")
inputs, outputs, program = module.context(trainable=False)
vocab = load_vocab(module.get_vocab_path())

word_ids = inputs["word_ids"]
embedding = outputs["word_embs"]

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[word_ids], place=place)

for item in processed_data:
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
```
