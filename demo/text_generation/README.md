# PaddleHub 文本生成

本示例将展示如何使用PaddleHub Fine-tune API以及Transformer类预训练模型(ERNIE/BERT/RoBERTa)完成生成任务。

## 目录结构
```
text_generation
├── predict.py # 预测脚本
├── README.md # 文本生成迁移学习文档说明
├── run_text_gen.sh # 训练启动脚本
├── run_predict.sh # # 预测启动脚本
├── text_gen.py # 训练脚本
```

## 如何开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_text_gen.sh`即可开始使用ERNIE对Couplet数据集进行Fine-tune。

其中脚本参数说明如下：

```bash
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；
--learning_rate: Fine-tune的最大学习率；
--cut_fraction: Slanted triangular策略中学习率上升阶段的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为learning_rate/32；
--num_epoch: Fine-tune迭代的轮数；
--max_seq_len: ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；
--use_data_parallel: 是否使用并行计算，默认True。打开该功能依赖nccl库；
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型；
```

## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

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

如果想尝试BERT模型，只需要更换Module中的`name`参数即可.
```python
# 更换name参数即可无缝切换BERT中文模型, 代码示例如下
module = hub.Module(name="bert_chinese_L-12_H-768_A-12")
```

### Step2: 准备数据集并使用tokenizer预处理数据
```python
tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())
dataset = hub.dataset.Couplet(
    tokenizer=tokenizer, max_seq_len=128)
```
**NOTE**:
* 即使是使用ernie_tiny预训练模型，也请使用BertTokenizer，而不要使用ErnieTinyTokenizer。因为对联数据集中上联是按字切分并以特殊字符"\002"作为分隔符的。

数据集的准备代码可以参考 [couplet.py](../../paddlehub/dataset/couplet.py)。

`hub.dataset.Couplet()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录；

`module.get_vocab_path()` 会返回预训练模型对应的词表；

`max_seq_len` 需要与Step1中context接口传入的序列长度保持一致；

dataset将调用传入的tokenizer提供的encode接口对全量数据进行预处理，您可以通过以下方式观察数据的处理流程：
```
single_result = tokenizer.encode(text="hello", text_pair="world", max_seq_len=10) # BertTokenizer
# {'input_ids': [3, 1, 5, 39825, 5, 0, 0, 0, 0, 0], 'segment_ids': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 'seq_len': 5, 'input_mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
dataset_result = dataset.get_dev_records() # set dataset max_seq_len = 10
# {'input_ids': [101, 1745, 1751, 100, 100, 100, 100, 100, 100, 102], 'segment_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'seq_len': 10, 'input_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'label': [144, 643, 7057, 12130, 8375, 296, 525, 1884, 702, 702], 'dec_input': [134, 144, 643, 7057, 12130, 8375, 296, 525, 1884, 702]}
```

#### 自定义数据集

如果想加载自定义数据集完成迁移学习，详细参见[自定义数据集](../../docs/tutorial/how_to_load_data.md)。

### Step3：选择优化策略和运行配置

```python
strategy = hub.ULMFiTStrategy(
    learning_rate=5e-3,
    optimizer_name="adam",
    cut_fraction=0.1,
    dis_params_layer=module.get_params_layer(),
    frz_params_layer=module.get_params_layer())

config = hub.RunConfig(use_cuda=True, num_epoch=3, batch_size=32, strategy=strategy)
```

#### 优化策略
由于文本生成任务不是简单拼接全连接层即可完成，它引入了RNN seq2seq结构，带来了较多的新参数。得益于PaddleHub的ULMFiT策略，我们可以用较大的学习率更新Decoder中的参数，同时减缓Encoder中的参数更新速度。实验表明，在本任务中使用ULMFiTStrategy可以取得比AdamWeightDecayStrategy更好的效果。

* `learning_rate`: Fine-tune过程中的最大学习率；
* `optimizer_name`： 优化器类别。
* `cut_fraction`: Slanted triangular策略中学习率上升阶段的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为learning_rate/32；
* `dis_params_layer`: 分层学习率策略需要的参数层次信息，如果设置为module.get_params_layer()，预训练模型中各层神经网络的更新速度将逐层衰减，默认每一层的学习率是上一层学习率的1/2.6；
* `frz_params_layer`: 逐层解冻策略需要的参数层次信息，如果设置为module.get_params_layer()，预训练模型中各层神经网络将在训练过程中随着epoch的增大而参与更新，例如epoch=1时只有最上层参数会更新，epoch=2时最上2层参数都会参与更新；

关于ULMFiT策略的详细说明，请参考[论文](https://arxiv.org/pdf/1801.06146.pdf)。如果您希望将ULMFiT策略与AdamWeightDecay策略进行组合实验，请参考[CombinedStrategy](../../docs/reference/strategy.md)

#### 运行配置
`RunConfig` 主要控制Fine-tune的训练，包含以下可控制的参数:

* `use_cuda`: 是否使用GPU训练，默认为False；
* `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
* `num_epoch`: Fine-tune的轮数；
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* `strategy`: Fine-tune优化策略；

### Step4: 构建网络并创建生成迁移任务进行Fine-tune
```python
pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]

gen_task = hub.TextGenerationTask(
    dataset=dataset,
    feature=pooled_output,
    token_feature=sequence_output,
    max_seq_len=128,
    num_classes=dataset.num_labels,
    config=config,
    metrics_choices=["bleu"])

gen_task.finetune_and_eval()
```
**NOTE:**
1. `outputs["pooled_output"]`返回了Transformer类预训练模型对应的[CLS]向量,可以用于句子或句对的特征表达。这一特征将用于TextGenerationTask Decoder状态初始化。
2. `outputs["sequence_output"]`返回了ERNIE/BERT模型输入单词的对应输出,可以用于单词的特征表达；这一特征将用于TextGenerationTask Decoder解码。
3. 当前TextGenerationTask采用如下图所示的seq2seq结构：

<p align="center">
 <img src="https://d2l.ai/_images/encoder-decoder.svg" width='60%' align="middle"
</p>

其中Encoder为hub.Module指定的预训练模型，Decoder为通用的LSTM+Attention结构.

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
CKPT_DIR="ckpt_generation/"
python predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 128
```
其中CKPT_DIR为Fine-tune API保存最佳模型的路径, max_seq_len是ERNIE模型的最大序列长度，*请与训练时配置的参数保持一致*

参数配置正确后，请执行脚本`sh run_predict.sh`，即可看到文本生成预测结果。

我们在AI Studio上提供了IPython NoteBook形式的demo，点击[PaddleHub教程合集](https://aistudio.baidu.com/aistudio/projectdetail/231146)，可使用AI Studio平台提供的GPU算力进行快速尝试。


## 超参优化AutoDL Finetuner

PaddleHub还提供了超参优化（Hyperparameter Tuning）功能， 自动搜索最优模型超参得到更好的模型效果。详细信息参见[AutoDL Finetuner超参优化功能教程](../../docs/tutorial/autofinetune.md)。


## Fine-tune之后保存的模型转化为PaddleHub Module

Fine-tune之后保存的模型转化为PaddleHub Module[教程](../../docs/tutorial/finetuned_model_to_module.md)
