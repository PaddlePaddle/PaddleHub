## 模型概述

主题模型(Topic Model)是以无监督学习的方式对文档的隐含语义结构进行聚类的统计模型，其中SLDA(Sentence-LDA)是主题模型的一种。SLDA是LDA主题模型的扩展，LDA假设每个单词对应一个主题，而SLDA假设每个句子对应一个主题。本Module基于的数据集为百度自建的小说领域数据集。

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

## 查看代码
https://github.com/baidu/Familia


## 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.8.0


## 更新历史

* 1.0.0

  初始发布
