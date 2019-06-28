# 如何使用PaddleHub完成文本分类迁移

文本分类迁移是NLP迁移学习中最常见的一个任务之一。教程以情感分析任务为例子，介绍下如何使用PaddleHub+Fine-tune API快速完成文本分类迁移任务。

## 教程前置条件

* 已完成PaddlePaddle和PaddleHub的安装
* 对BERT/ERNIE等Transformer类模型有基本的了解


## ERNIE介绍

ERNIE是百度开放的基于Transformer知识增强的语义表示模型（**E**nhanced **R**epresentation from k**N**owledge **I**nt**E**gration）ERNIE预训练模型结合Fine-tuning，可以在中文情感分析任务上可以得到非常不错的效果。更多的介绍可以参考[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)

## 快速开始

### 准备环境
```python
import paddle.fluid as fluid
import paddlehub as hub
```
### 加载预训练模型

通过PaddleHub，只需要一行代码，即可以获取到PaddlePaddle生态下的预训练模型。

```python
module = hub.Module(name="ernie")

inputs, outputs, program = module.context(trainable="True", max_seq_len=128)
```

* 通过`hub.Module(name="ernie")`PaddleHub会自动下载并加载ERNIE模型。
* `module.context`接口中，`trainable=True`则预训练模型的参数可以被训练，`trainble=False`则讲预训练模型参数不可修改，仅作为特征提取器使用。
* `max_seq_len`是ERNIE/BERT模型特有的参数，控制模型最大的序列识别长度，这一参数与任务相关，如果显存有限，切任务文本长度较短，可以适当调低这一参数。如果处理文本序列的unicode字符长度超过`max_seq_len`，则模型会对序列进行截断。通常来说，128是一个性能均衡的默认值。

#### ERNIE的输入输出结构

ERNIE模型与BERT在结构上类似，如下图所示：
![ERNIE结构图](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v0.5.0/docs/imgs/ERNIE_input_output.png)

ERNIE的在PaddleHub中的的输入有4个Tensor，分别是：
* `input_ids`: 文本序列后切词的ID；
* `position_ids`: 文本序列的位置ID；
* `segment_ids`: 文本序列的类型；
* `input_mask`: 序列的mask信息，主要用于对padding的标识；

前三个输入与BERT模型的论文输入对应，第四个输入为padding所需的标识信息。更多细节信息可参考论文[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

ERNIE的输出的话有两类，分别是

* `pooled_output`: 句子粒度特征，对应的shape为`[batch_size, hidden_size]`，可用于句子分类或句对分类任务。
* `sequence_output`: 词粒度的特征，对应的shape为`[batch_size, max_seq_len, hidden_size]`, 可用于序列标注任务。

通过以下代码即可获取到对应特征的Tensor，可用于后续的组网工作。
```python
inputs, outputs, program = module.context(trainable="True", max_seq_len=128)

pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]
```

### 准备数据及数据预处理

```python
ds = hub.dataset.ChnSentiCorp()
reader = hub.reader.ClassifyReader(dataset=ds, vocab_path=module.get_vocab_path(), max_seq_len=128)
```

通过`hub.dataset.ChnSentiCorp()`会自动获取数据集，可以通过以下代码查看训练集中的文本与标注：
```python
ds = hub.dataset.ChnSentiCorp()
for e in ds.get_train_examples():
	print(e.text_a, e.label)
```


`ClassifyReader`是专门ERNIE/BERT模型的数据预处理器，会根据模型词典，进行字粒度的切词，其中英文以词粒度进行分割，而中文和其他字符采用unicode为单位的字粒度切词。因此与传统的中文分词器有所区别，详细代码可以参考 [tokenization.py](https://github.com/PaddlePaddle/PaddleHub/blob/release/v0.5.0/paddlehub/reader/tokenization.py)

`ClassifyReader`的参数有以下三个：
* `dataset`: 传入PaddleHub Dataset;
* `vocab_path`: 传入ERNIE/BERT模型对应的词表文件路径;
* `max_seq_len`: ERNIE模型的最大序列长度，若序列长度不足，会通过padding方式补到`max_seq_len`, 若序列长度大于该值，则会以截断方式让序列长度为`max_seq_len`;


### 创建迁移学习任务

```python
task = hub.create_text_cls_task(feature=pooled_output, num_classes=ds.num_labels)
```

### 配置优化策略

适用于ERNIE/BERT这类Transformer模型的迁移优化策略为`AdamWeightDecayStrategy`
```python
strategy=hub.AdamWeightDecayStrategy(
    learning_rate=1e-4,
    lr_scheduler="linear_decay",
    warmup_proportion=0.0,
    weight_decay=0.01
)
```
* `learning_rate`: 最大学习率
* `lr_scheduler`: 有`linear_decay`和`noam_decay`两种衰减策略可选, 如下图所示：
* `warmup_proprotion`: 训练预热的比例，若设置为0.1, 则会在前10%的训练step中学习率逐步提升到`learning_rate`
* `weight_decay`: 权重衰减，类似模型正则项策略，避免模型overfitting

![学习率衰减策略](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v0.5.0/docs/imgs/decay_strategy.png)

### 设置运行配置

关于运行配置的详细信息可以查看[RunConfig](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/api/RunConfig.md)

```python
config = hub.RunConfig(
    use_cuda=True,
    num_epoch=3,
    batch_size=32,
    strategy=strategy)
```
请根据实际情况选择`use_cuda`和`batch_size`，若出现显存不足的情况，请调低`batch_size`

### feed_list
```python
feed_list = [
    inputs["input_ids"].name, inputs["position_ids"].name,
    inputs["segment_ids"].name, inputs["input_mask"].name,
    task.variable("label").name
]
```
`feed_list`的配置需要与ClassifyReader的Tensor输出顺序保持一致。请注意，此处的tensor name顺序不可以改变，因为`ClassifyReader`就是按照这一顺序返回ERNIE所需的输入tensor。

### 启动Fine-tuning

当配置好迁移任务、数据预处理、`FeedList`和`RunConfig`后，就可以使用`finetune_and_eval`启动Fine-tuning任务了

```python
hub.finetune_and_eval(
    task=cls_task,
    data_reader=reader,
    feed_list=feed_list,
    config=config)
```

`finetune_and_eval`接口会自动完成验证集评估，并保存最优模型，并自动完成Visual DL可视化，如下图所示，在`/path/to/ckpt/vdllog`中会保存Visual DL的打点信息，如下图所示启动Visual DL后即可看到Fine-tuning的变化过程, 包括Loss变化，训练集和验证集的准确率情况等。
![VisualDL可视化](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v0.5.0/docs/imgs/finetune_vdl.png)
