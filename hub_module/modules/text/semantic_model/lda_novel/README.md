## 模型概述

本Module基于的数据集为百度自建的小说领域数据集。主题模型(Topic Model)是以无监督学习的方式对文档的隐含语义结构进行聚类的统计模型，其中LDA(Latent Dirichlet Allocation)算法是主题模型的一种。LDA根据对词的共现信息的分析，拟合出词-文档-主题的分布，从而将词、文本映射到一个语义空间中。

更多详情请参考[LDA论文](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)。

注：该Module由第三方开发者DesmonDay贡献。

## LDA模型 API 说明
### cal_doc_distance(doc_text1, doc_text2)
用于计算两个输入文档之间的距离，包括Jensen-Shannon divergence(JS散度)、Hellinger Distance(海林格距离)。

**参数**

- doc_text1(str): 输入的第一个文档。
- doc_text2(str): 输入的第二个文档。

**返回**

- jsd(float): 两个文档之间的JS散度。
- hd(float): 两个文档之间的海林格距离。

### cal_doc_keywords_similarity(document, top_k=10)

用于查找输入文档的前k个关键词。

**参数**

- document(str): 输入文档。
- top_k(int): 查找输入文档的前k个关键词。

**返回**

- results(list): 包含每个关键词以及对应的与原文档的相似度。其中，list的基本元素为dict，dict的key为关键词，value为对应的与原文档的相似度。

### cal_query_doc_similarity(query, document)

用于计算短文档与长文档之间的相似度。

**参数**

- query(str): 输入的短文档。
- document(str): 输入的长文档。

**返回**

- lda_sim(float): 返回短文档与长文档之间的相似度。

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

这里展示部分API的使用示例。
``` python
import paddlehub as hub

lda_novel = hub.Module(name="lda_novel")
jsd, hd = lda_novel.cal_doc_distance(doc_text1="老人幸福地看着自己的儿子，露出了欣慰的笑容。", doc_text2="老奶奶看着自己的儿子，幸福地笑了。")
# jsd = 0.01292, hd = 0.11893

lda_sim = lda_novel.cal_query_doc_similarity(query='亲孙女', document='老人激动地打量着面前的女孩，似乎找到了自己的亲孙女一般，双手止不住地颤抖着。')
# LDA similarity = 0.0

```

## 查看代码
https://github.com/baidu/Familia


## 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.8.0

## 更新历史

* 1.0.0

  初始发布
