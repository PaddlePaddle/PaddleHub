# ERNIE Classification

本示例将展示如何使用PaddleHub Finetune API利用ERNIE完成分类任务。

其中分类任务可以分为两大类

* 单句分类
  - 中文情感分析任务 ChnSentiCorp


* 句对分类
  - 语义相似度 LCQMC
  - 检索式问答任务 NLPCC-DBQA

## 如何开始Finetune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_sentiment_cls.sh`即可开始使用ERNIE对ChnSentiCorp数据集进行Finetune。

其中脚本参数说明如下：

```bash
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足错误，请调低这一参数值
--weight_decay:
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型
--num_epoch: Finetune迭代的轮数
--max_seq_len: ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足错误，请调低这一参数
```

## 代码步骤

使用PaddleHub Finetune API进行Finetune可以分为一下4个步骤

### Step1: 加载预训练模型

```python
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
```
其中最大序列长度`max_seq_len`是可以调整的参数，建议值128，根据任务文本长度不同可以调整该值，但最大不超过512。

如果想尝试BERT模型，例如BERT中文模型，只需要更换Module中的参数即可.
PaddleHub除了ERNIE，还提供以下BERT模型:

BERT模型名                         | PaddleHub Module
---------------------------------- | :------:
BERT-Base, Uncased                 | `hub.Module(name='bert_uncased_L-12_H-768_A-12')`
BERT-Large, Uncased                | `hub.Module(name='bert_uncased_L-24_H-1024_A-16')`
BERT-Base, Cased                   | `hub.Module(name='bert_cased_L-12_H-768_A-12')`
BERT-Large, Cased                  | `hub.Module(name='bert_cased_L-24_H-1024_A-16')`
BERT-Base, Multilingual Cased      | `hub.Module(nane='bert_multi_cased_L-12_H-768_A-12')`
BERT-Base, Chinese                 | `hub.Module(name='bert_chinese_L-12_H-768_A-12')`


```python
# 更换name参数即可无缝切换BERT中文模型
module = hub.Module(name="bert_chinese_L-12_H-768_A-12")
```

### Step2: 准备数据集并使用ClassifyReader读取数据
```python
reader = hub.reader.ClassifyReader(
    dataset=hub.dataset.ChnSentiCorp(),
    vocab_path=module.get_vocab_path(),
    max_seq_len=128)
```
`hub.dataset.ChnSentiCorp()` 会自动从网络下载数据集并解压到用户目录下.paddlehub/dataset目录

`module.get_vaocab_path()` 会返回ERNIE/BERT模型对应的词表

`max_seq_len`需要与Step1中context接口传入的序列长度保持一致

ClassifyReader中的`data_generator`会自动按照模型对应词表对数据进行切词，以迭代器的方式返回ERNIE/BERT所需要的Tensor格式，包括`input_ids`，`position_ids`，`segment_id`与序列对应的mask `input_mask`.


### Step3: 构建网络并创建分类迁移任务
```python
with fluid.program_guard(program): # NOTE: 必须使用fluid.program_guard接口传入Module返回的预训练模型program
    label = fluid.layers.data(name="label", shape=[1], dtype='int64')

    pooled_output = outputs["pooled_output"]

    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name, label.name
    ]

    cls_task = hub.create_text_classification_task(
        feature=pooled_output, label=label, num_classes=reader.get_num_labels())
```
**NOTE:** 基于预训练模型的迁移学习网络搭建，必须在`with fluid.program_gurad()`作用域内组件网络
1. `outputs["pooled_output"]`返回了ERNIE/BERT模型对应的[CLS]向量,可以用于句子或句对的特征表达。
2. `feed_list`中的inputs参数指名了ERNIE/BERT中的输入tensor，以及labels顺序，与ClassifyReader返回的结果一致。
3. `create_text_classification_task`通过输入特征，label与迁移的类别数，可以生成适用于文本分类的迁移任务`cls_task`

### Step4：选择优化策略并开始Finetune

```python
strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    learning_rate=5e-5,
    warmup_strategy="linear_warmup_decay",
)

config = hub.RunConfig(use_cuda=True, num_epoch=3, batch_size=32, strategy=strategy)

hub.finetune_and_eval(task=cls_task, data_reader=reader, feed_list=feed_list, config=config)
```
针对ERNIE与BERT类任务，PaddleHub封装了适合这一任务的迁移学习优化策略。用户可以通过配置学习率，权重
