## 模型概述

主题模型(Topic Model)是以无监督学习的方式对文档的隐含语义结构进行聚类的统计模型，其中SLDA(Sentence-LDA)是主题模型的一种。SLDA是LDA主题模型的扩展，LDA假设每个单词对应一个主题，而SLDA假设每个句子对应一个主题。本Module基于的数据集为百度自建的微博领域数据集。

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/nlp/semantic_model/slda.png" hspace='10'/> <br />
</p>

更多详情请参考[SLDA论文](https://pdfs.semanticscholar.org/c311/778adb9484c86250e915aecd9714f4206050.pdf)。

注：该Module由第三方开发者DesmonDay贡献。

## SLDA模型 API 说明

### infer_doc_topic_distribution(document)

用于推理出文档的主题分布。

**参数**

- document(str): 输入文档。

**返回**

- results(list): 包含主题分布下各个主题ID和对应的概率分布。其中，list的基本元素为dict，dict的key为主题ID，value为各个主题ID对应的概率。

### show_topic_keywords(topic_id, k=10)

用于展示出每个主题下对应的关键词，可配合推理主题分布的API使用。

**参数**

- topic_id(int): 主题ID。
- k(int): 需要知道对应主题的前k个关键词。

**返回**

- results(dict): 返回对应文档的前k个关键词，以及各个关键词在文档中的出现概率。

### 代码示例

这里展示API的使用示例。

``` python
import paddlehub as hub

slda_weibo = hub.Module(name="slda_weibo")

topic_dist = slda_weibo.infer_doc_topic_distribution("百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。")
# [{'topic id': 874, 'distribution': 0.5}, {'topic id': 1764, 'distribution': 0.5}]

keywords = slda_weibo.show_topic_keywords(topic_id=874)
# {'数据': 0.07850538018570305,
#  '更新': 0.04504777051711974,
#  '出口': 0.023363758946167185,
#  '信息': 0.020567061200812687,
#  '全国': 0.015975367546781145,
#  '双十一': 0.014998636225687216,
#  '地理': 0.013257422965959297,
#  '官方': 0.012913598174463106,
#  '支持': 0.01177359809763076,
#  '说话': 0.011205999070328388}

```
## 查看代码
https://github.com/baidu/Familia


## 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.8.0

## 更新历史

* 1.0.0

  初始发布
