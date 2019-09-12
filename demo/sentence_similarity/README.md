# Word2vec

## 关于

本示例展示如何使用word2vec_skipgram Module进行句子相似度预测。

word2vec_skipgram是对中文词语的向量表示，可以用于各类NLP下游任务等。

## 准备工作

在运行本目录的脚本前，需要先安装1.4.0版本以上的PaddlePaddle（如果您本地已经安装了符合条件的PaddlePaddle版本，那么可以跳过`准备工作`这一步）。

```shell
# 安装GPU版本的PaddlePaddle
$ pip install --upgrade paddlepaddle-gpu
```

如果您的机器不支持GPU，可以通过下面的命令来安装CPU版本的PaddlePaddle

```shell
# 安装CPU版本的PaddlePaddle
$ pip install --upgrade paddlepaddle
```

在安装过程中如果遇到问题，您可以到[Paddle官方网站](http://www.paddlepaddle.org/)上查看解决方案。

## 预测

```shell
python sensim.py
```

程序运行结束后, 可以看待预测的两个文本的余弦相似度

```
text_a: 驾驶 违章 一次 扣 12分 用 两个 驾驶证 处理 可以 吗; text_b:  一次性 扣 12分 的 违章 , 能用 不满 十二分 的 驾驶证 扣分 吗; cosine_similarity: 0.39889
text_a: 水果 放 冰箱 里 储存 好 吗; text_b: 中国银行 纪念币 网上 怎么 预约; cosine_similarity: -0.08258
text_a: 电脑 反应 很 慢 怎么 办; text_b: 反应 速度 慢 , 电脑 总是 卡 是 怎么回事; cosine_similarity: 0.40820
```
