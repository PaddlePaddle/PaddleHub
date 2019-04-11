# ERNIE Classification

本示例将展示如何使用PaddleHub Finetune API利用ERNIE完成分类任务。

其中分类任务可以分为两大类

* 单句分类
- 中文情感分析任务 ChnSentiCorp


* 句对分类
- 语义相似度 LCQMC
- 检索式问答任务 nlpcc-dbqa

## 如何开始Finetune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_sentiment_cls.sh`即可开始使用ERNIE对ChnSentiCorp数据集进行Finetune。

其中脚本参数说明如下：

```bash
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足错误，请调低这一参数值
--weight_decay:
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型
--num_epoch: Finetune迭代的轮数
--max_seq_len: ERNIE模型使用的最大序列长度，最大不能超过512,
  若出现显存不足错误，请调低这一参数
```

## 代码步骤

使用PaddleHub Finetune API进行Finetune可以分为一下4个步骤

### Step1: 加载预训练模型

```python
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=128)
```
其中最大序列长度`max_seq_len`是可以调整的参数，建议值128，根据任务文本长度不同可以调整该值，但最大不超过512。

如果想尝试BERT模型，例如BERT中文模型，只需要更换Module中的参数即可.
PaddleHub除了ERNIE，还提供以下BERT模型:

BERT模型名                         | PaddleHub Module name
---------------------------------- | :------:
BERT-Base, Uncased                 | bert_uncased_L-12_H-768_A-12
BERT-Large, Uncased                | bert_uncased_L-24_H-1024_A-16
BERT-Base, Cased                   | bert_cased_L-12_H-768_A-12
BERT-Large, Cased                  | bert_cased_L-24_H-1024_A-16
BERT-Base, Multilingual Cased      | bert_multi_cased_L-12_H-768_A-12
BERT-Base, Chinese                 | bert_chinese_L-12_H-768_A-12


```python
    # 即可无缝切换BERT中文模型
    module = hub.Module(name="bert_chinese_L-12_H-768_A-12")
```
