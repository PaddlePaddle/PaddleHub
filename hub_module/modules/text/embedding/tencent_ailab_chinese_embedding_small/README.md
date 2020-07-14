## 概述

Tencent_AILab_ChineseEmbedding提供了基于海量中文语料训练学习得到的800多万个中文词语和短语的词向量表示，每一个词向量为200维。
该Module截取了原来词汇表中前200万的词语，同样可以用于各种下游任务迁移学习。

更多详情参考: https://ai.tencent.com/ailab/nlp/en/embedding.html

## API

```python
def context(trainable=False, max_seq_len=128, num_slots=1)
```

获取该Module的预训练program以及program相应的输入输出。

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变。
* max_seq_len(int): 模型使用的最大序列长度。
* num_slots(int): 输入到模型所需要的文本个数，如完成单句文本分类任务，则num_slots=1；完成pointwise文本匹配任务，则num_slots=2；完成pairtwise文本匹配任务，则num_slots=3；

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

### 代码示例

```python
import paddlehub as hub
import cv2

tencent_ailab_chinese_embedding = hub.Module(name="tencent_ailab_chinese_embedding_small")
inputs, outputs, program = tencent_ailab_chinese_embedding.context(trainable=True, max_seq_len=128, num_slots=1)
```

## 依赖

paddlepaddle >= 1.8.0

paddlehub >= 1.8.0

## 更新历史

* 1.0.0

  初始发布
