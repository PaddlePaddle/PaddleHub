## 概述

Word2vec是常用的词嵌入（word embedding）模型。该PaddleHub Module基于Skip-gram模型，在海量百度搜索数据集下预训练得到中文单词预训练词嵌入。其支持Fine-tune。Word2vec的预训练数据集的词汇表大小为1700249，word embedding维度为128。

## API

### context(trainable=False, max_seq_len=128, num_slots=1)

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

# Load word2vec pretrained model
module = hub.Module(name="word2vec_skipgram")
inputs, outputs, program = module.context(trainable=True)

# Must feed all the tensor of module need
word_ids = inputs["text"]

# Use the pretrained word embeddings
embedding = outputs["emb"]
```

## 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.8.0

## 更新历史

* 1.0.0

  初始发布

* 1.1.0

  模型升级，支持用于文本分类，文本匹配等各种任务迁移学习
