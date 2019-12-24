# 句子相似度

本示例展示如何使用word2vec_skipgram模型进行句子相似度预测。



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
