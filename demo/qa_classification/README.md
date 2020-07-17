# PaddleHub 检索式问答任务

本示例将展示如何使用PaddleHub Fine-tune API以及Transformer类预训练模型在NLPCC-DBQA数据集上完成检索式问答任务。

## 如何开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_classifier.sh`即可开始使用ERNIE对NLPCC-DBQA数据集进行Fine-tune。

其中脚本参数说明如下：

```bash
# 模型相关
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；
--learning_rate: Fine-tune的最大学习率；
--weight_decay: 控制正则项力度的参数，用于防止过拟合，默认为0.01；
--warmup_proportion: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0；
--num_epoch: Fine-tune迭代的轮数；
--max_seq_len: ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；

# 任务相关
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型；
```

## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤：

### Step1: 加载预训练模型

```python
import paddlehub as hub

module = hub.Module(name="ernie")
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
```
其中最大序列长度`max_seq_len`是可以调整的参数，建议值128，根据任务文本长度不同可以调整该值，但最大不超过512。

PaddleHub还提供BERT等模型可供选择, 模型对应的加载示例如下：

   模型名                           | PaddleHub Module
---------------------------------- | :------:
ERNIE, Chinese                     | `hub.Module(name='ernie')`
ERNIE tiny, Chinese                | `hub.Module(name='ernie_tiny')`
ERNIE 2.0 Base, English            | `hub.Module(name='ernie_v2_eng_base')`
ERNIE 2.0 Large, English           | `hub.Module(name='ernie_v2_eng_large')`
BERT-Base, Uncased                 | `hub.Module(name='bert_uncased_L-12_H-768_A-12')`
BERT-Large, Uncased                | `hub.Module(name='bert_uncased_L-24_H-1024_A-16')`
BERT-Base, Cased                   | `hub.Module(name='bert_cased_L-12_H-768_A-12')`
BERT-Large, Cased                  | `hub.Module(name='bert_cased_L-24_H-1024_A-16')`
BERT-Base, Multilingual Cased      | `hub.Module(nane='bert_multi_cased_L-12_H-768_A-12')`
BERT-Base, Chinese                 | `hub.Module(name='bert_chinese_L-12_H-768_A-12')`
BERT-wwm, Chinese                  | `hub.Module(name='bert_wwm_chinese_L-12_H-768_A-12')`
BERT-wwm-ext, Chinese              | `hub.Module(name='bert_wwm_ext_chinese_L-12_H-768_A-12')`
RoBERTa-wwm-ext, Chinese           | `hub.Module(name='roberta_wwm_ext_chinese_L-12_H-768_A-12')`
RoBERTa-wwm-ext-large, Chinese     | `hub.Module(name='roberta_wwm_ext_chinese_L-24_H-1024_A-16')`

更多模型请参考[PaddleHub官网](https://www.paddlepaddle.org.cn/hub?filter=hot&value=1)。

如果想尝试BERT模型，只需要更换Module中的`name`参数即可。
```python
# 更换name参数即可无缝切换BERT中文模型, 代码示例如下
module = hub.Module(name="bert_chinese_L-12_H-768_A-12")
```

### Step2: 准备数据集并使用tokenizer预处理数据
```python
tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())
dataset = hub.dataset.NLPCC_DBQA(
        tokenizer=tokenizer, max_seq_len=128)
```
如果是使用ernie_tiny预训练模型，请使用ErnieTinyTokenizer。
```
tokenizer = hub.ErnieTinyTokenizer(
    vocab_file=module.get_vocab_path(),
    spm_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path())
```
ErnieTinyTokenizer和BertTokenizer的区别在于它将按词粒度进行切分，详情请参考[文章](https://www.jiqizhixin.com/articles/2019-11-06-9)。

数据集的准备代码可以参考[nlpcc_dbqa.py](../../paddlehub/dataset/nlpcc_dbqa.py)。

`hub.dataset.NLPCC_DBQA())` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录；

`module.get_vocab_path()` 会返回预训练模型对应的词表；

`module.sp_model_path` 若module为ernie_tiny则返回对应的子词切分模型路径，否则返回None；

`module.word_dict_path` 若module为ernie_tiny则返回对应的词语切分模型路径，否则返回None；

`max_seq_len` 需要与Step1中context接口传入的序列长度保持一致；

dataset将调用传入的tokenizer提供的encode接口对全量数据进行预处理，您可以通过以下方式观察数据的处理流程：
```
single_result = tokenizer.encode(text="hello", text_pair="world", max_seq_len=10) # BertTokenizer
# {'input_ids': [3, 1, 5, 39825, 5, 0, 0, 0, 0, 0], 'segment_ids': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 'seq_len': 5, 'input_mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
dataset_result = dataset.get_dev_records() # set dataset max_seq_len = 10
# {'input_ids': [3, 1637, 964, 1676, 613, 5, 1637, 964, 1676, 5], 'segment_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 'seq_len': 10, 'input_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'label': 0}
```

#### 自定义数据集

如果想加载自定义数据集完成迁移学习，详细参见[自定义数据集](../../docs/tutorial/how_to_load_data.md)。

### Step3：选择优化策略和运行配置

```python
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_proportion=0.0,
    lr_scheduler="linear_decay",
)

config = hub.RunConfig(use_cuda=True, use_data_parallel=True, use_pyreader=True, num_epoch=3, batch_size=32, strategy=strategy)
```

#### 优化策略

PaddleHub提供了许多优化策略，如`AdamWeightDecayStrategy`、`ULMFiTStrategy`、`DefaultFinetuneStrategy`等，详细信息参见[策略](../../docs/reference/strategy.md)。

针对ERNIE与BERT类任务，PaddleHub封装了适合这一任务的迁移学习优化策略`AdamWeightDecayStrategy`：

* `learning_rate`: Fine-tune过程中的最大学习率；
* `weight_decay`: 模型的正则项参数，默认0.01，如果模型有过拟合倾向，可适当调高这一参数；
* `warmup_proportion`: 如果warmup_proportion>0, 例如0.1, 则学习率会在前10%的steps中线性增长至最高值learning_rate；
* `lr_scheduler`: 有两种策略可选(1) `linear_decay`策略学习率会在最高点后以线性方式衰减; `noam_decay`策略学习率会在最高点以多项式形式衰减；

#### 运行配置
`RunConfig` 主要控制Fine-tune的训练，包含以下可控制的参数：

* `log_interval`: 进度日志打印间隔，默认每10个step打印一次；
* `eval_interval`: 模型评估的间隔，默认每100个step评估一次验证集；
* `save_ckpt_interval`: 模型保存间隔，请根据任务大小配置，默认只保存验证集效果最好的模型和训练结束的模型；
* `use_cuda`: 是否使用GPU训练，默认为False；
* `use_pyreader`: 是否使用pyreader，默认False；
* `use_data_parallel`: 是否使用并行计算，默认False。打开该功能依赖nccl库；
* `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
* `num_epoch`: Fine-tune的轮数；
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* `enable_memory_optim`: 是否使用内存优化， 默认为True；
* `strategy`: Fine-tune优化策略；

### Step4: 构建网络并创建分类迁移任务进行Fine-tune
```python
pooled_output = outputs["pooled_output"]

cls_task = hub.TextClassifierTask(
    dataset=dataset,
    feature=pooled_output,
    num_classes=dataset.num_labels,
    config=config)

cls_task.finetune_and_eval()
```
**NOTE:**
1. `outputs["pooled_output"]`返回了ERNIE/BERT模型对应的[CLS]向量,可以用于句子或句对的特征表达。
2. `hub.TextClassifierTask`通过输入特征，label与迁移的类别数，可以生成适用于文本分类的迁移任务`TextClassifierTask`

#### 自定义迁移任务

如果想改变迁移任务组网，详细参见[自定义迁移任务](../../docs/tutorial/how_to_define_task.md)。

## 可视化

Fine-tune API训练过程中会自动对关键训练指标进行打点，启动程序后执行下面命令：
```bash
$ visualdl --logdir $CKPT_DIR/visualization --host ${HOST_IP} --port ${PORT_NUM}
```
其中${HOST_IP}为本机IP地址，${PORT_NUM}为可用端口号，如本机IP地址为192.168.0.1，端口号8040，用浏览器打开192.168.0.1:8040，即可看到训练过程中指标的变化情况。

## 模型预测

通过Fine-tune完成模型训练后，在对应的ckpt目录下，会自动保存验证集上效果最好的模型。
配置脚本参数
```
CKPT_DIR="./ckpt_qa"
python predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 128
```
其中CKPT_DIR为Fine-tune API保存最佳模型的路径, max_seq_len是ERNIE模型的最大序列长度，*请与训练时配置的参数保持一致*。

参数配置正确后，请执行脚本`sh run_predict.sh`，即可看到以下文本分类预测结果, 以及最终准确率。
如需了解更多预测步骤，请参考`predict.py`。

我们在AI Studio上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|ResNet|图像分类|猫狗数据集DogCat|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147010)||
|ERNIE|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147006)||
|ERNIE|文本分类|中文新闻分类数据集THUNEWS|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/221999)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成文本分类迁移学习。|
|ERNIE|序列标注|中文序列标注数据集MSRA_NER|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147009)||
|ERNIE|序列标注|中文快递单数据集Express|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/184200)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成序列标注迁移学习。|
|ERNIE Tiny|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/186443)||
|Senta|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216846)|本教程讲述了任何利用Senta和Fine-tune API完成情感分类迁移学习。|
|Senta|情感分析预测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215814)||
|LAC|词法分析|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215711)||
|Ultra-Light-Fast-Generic-Face-Detector-1MB|人脸检测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215962)||


## 超参优化AutoDL Finetuner

PaddleHub还提供了超参优化（Hyperparameter Tuning）功能， 自动搜索最优模型超参得到更好的模型效果。详细信息参见[AutoDL Finetuner超参优化功能教程](../../docs/tutorial/autofinetune.md)。
