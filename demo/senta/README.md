# Senta 情感分析

本示例展示如何使用PaddleHub Senta预训练模型进行预测。

Senta是百度NLP开放的中文情感分析模型，可以用于进行中文句子的情感分析，输出结果为`{正向/中性/负向}`中的一个，关于模型更多信息参见[Senta](https://www.paddlepaddle.org.cn/hubdetail?name=senta_bilstm&en_category=SentimentAnalysis), 本示例代码选择的是Senta-BiLSTM模型。

## 命令行方式预测

```shell
$ hub run senta_bilstm --input_text "这家餐厅很好吃"
$ hub run senta_bilstm --input_file test.txt
```

test.txt 存放待预测文本， 如：

```text
这家餐厅很好吃
这部电影真的很差劲
```

## 通过python API预测

`senta_demo.py`给出了使用python API调用Module预测的示例代码，
通过以下命令试验下效果。

```shell
python senta_demo.py
```

## 通过PaddleHub Fine-tune API微调
`senta_finetune.py` 给出了如何使用Senta模型的句子特征进行Fine-tuning的实例代码。
可以运行以下命令在ChnSentiCorp数据集上进行Fine-tuning。

```shell
$ sh run_finetune.sh
```

其中脚本参数说明如下：

```bash
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型；
--num_epoch: Fine-tune迭代的轮数；
--max_seq_len: 模型使用的最大序列长度， 若出现显存不足，请适当调低这一参数；
--use_gpu: 是否使用GPU进行训练，如果机器支持GPU且安装了GPU版本的PaddlePaddle，我们建议您打开这个开关；
```

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤：

### Step1: 加载预训练模型

```python
module = hub.Module(name="senta_bilstm")
inputs, outputs, program = module.context(trainable=True, max_seq_len=96)
```

其中最大序列长度`max_seq_len`是可以调整的参数，根据任务文本长度不同可以调整该值。

PaddleHub提供Senta一列模型可供选择, 模型对应的加载示例如下：

   模型名                           | PaddleHub Module
---------------------------------- | :------:
senta_bilstm                       | `hub.Module(name='senta_bilstm')`
senta_bow                          | `hub.Module(name='senta_bow')`
senta_gru                          | `hub.Module(name='senta_gru')`
senta_lstm                         | `hub.Module(name='senta_lstm')`
senta_cnn                          | `hub.Module(name='senta_cnn')`

更多模型请参考[PaddleHub官网](https://www.paddlepaddle.org.cn/hub?filter=hot&value=1)。

如果想尝GRU模型，只需要更换Module中的`name`参数即可。
```python
# 更换name参数即可无缝切换GRU模型, 代码示例如下
module = hub.Module(name="senta_gru")
```

### Step2: 选择Tokenizer读取数据

```python
tokenizer = hub.CustomTokenizer(
    vocab_file=module.get_vocab_path(),
    tokenize_chinese_chars=True,
)
```

`module.get_vocab_path()` 会返回预训练模型对应的词表；
`tokenize_chinese_chars` 是否切分中文文本

**NOTE:**
1. 如果使用Transformer类模型（如ERNIE、BERT、RoBerta等），则应该选择`hub.BertTokenizer`.
2. 如果使用非Transformer类模型（如senta、word2vec_skipgram、tencent_ailab_chinese_embedding_small等），则应该选择`hub.CustomTokenizer`

### Step3: 准备数据集
```python
dataset = hub.dataset.LCQMC(tokenizer=tokenizer, max_seq_len=128)
```

`hub.dataset.LCQMC()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录；

`max_seq_len` 需要与Step1中context接口传入的序列长度保持一致；

更多数据集信息参考[Dataset](../../docs/reference/dataset.md)。

#### 自定义数据集

如果想加载自定义数据集完成迁移学习，详细参见[自定义数据集](../../docs/tutorial/how_to_load_data.md)。

### Step3：选择优化策略和运行配置

```python
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_proportion=0.1,
    lr_scheduler="linear_decay",
)

config = hub.RunConfig(use_cuda=True, num_epoch=3, batch_size=32, strategy=strategy)
```

#### 优化策略

PaddleHub提供了许多优化策略，如`AdamWeightDecayStrategy`、`ULMFiTStrategy`、`DefaultFinetuneStrategy`等，详细信息参见[策略](../../docs/reference/strategy.md)。

其中`AdamWeightDecayStrategy`：

* `learning_rate`: Fine-tune过程中的最大学习率；
* `weight_decay`: 模型的正则项参数，默认0.01，如果模型有过拟合倾向，可适当调高这一参数；
* `warmup_proportion`: 如果warmup_proportion>0, 例如0.1, 则学习率会在前10%的steps中线性增长至最高值learning_rate；
* `lr_scheduler`: 有两种策略可选(1) `linear_decay`策略学习率会在最高点后以线性方式衰减; `noam_decay`策略学习率会在最高点以多项式形式衰减；

#### 运行配置
`RunConfig` 主要控制Fine-tune的训练，包含以下可控制的参数:

* `use_cuda`: 是否使用GPU训练，默认为False；
* `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
* `num_epoch`: Fine-tune的轮数；
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* `strategy`: Fine-tune优化策略；

### Step4: 构建网络并创建分类迁移任务进行Fine-tune
```python
sent_feature = outputs["sentence_feature"]

feed_list = [inputs["words"].name]

cls_task = hub.TextClassifierTask(
    dataset=dataset,
    feature=sent_feature,
    num_classes=2,
    config=config)

cls_task.finetune_and_eval()
```
**NOTE:**
1. `outputs["sentence_feature"]`返回了senta模型对应的句子特征,可以用于句子的特征表达；
2. `hub.TextClassifierTask`通过输入特征，label与迁移的类别数，可以生成适用于文本分类的迁移任务`TextClassifierTask`；

#### 自定义迁移任务

如果想改变迁移任务组网，详细参见[自定义迁移任务](../../docs/tutorial/how_to_define_task.md)。

## 可视化

Fine-tune API训练过程中会自动对关键训练指标进行打点，启动程序后执行下面命令
```bash
$ visualdl --logdir $CKPT_DIR/visualization --host ${HOST_IP} --port ${PORT_NUM}
```
其中${HOST_IP}为本机IP地址，${PORT_NUM}为可用端口号，如本机IP地址为192.168.0.1，端口号8040，用浏览器打开192.168.0.1:8040，即可看到训练过程中指标的变化情况。

## 模型预测

通过Fine-tune完成模型训练后，在对应的ckpt目录下，会自动保存验证集上效果最好的模型。
配置脚本参数
```
CKPT_DIR="ckpt_chnsentiment/"
python predict.py --checkpoint_dir $CKPT_DIR
```
其中CKPT_DIR为Fine-tune API保存最佳模型的路径

参数配置正确后，请执行脚本`sh run_predict.sh`，即可看到以下文本分类预测结果, 以及最终准确率。
如需了解更多预测步骤，请参考`predict.py`。

我们在AI Studio上提供了IPython NoteBook形式的demo，点击[PaddleHub教程合集](https://aistudio.baidu.com/aistudio/projectdetail/231146)，可使用AI Studio平台提供的GPU算力进行快速尝试。

## 超参优化AutoDL Finetuner

PaddleHub还提供了超参优化（Hyperparameter Tuning）功能， 自动搜索最优模型超参得到更好的模型效果。详细信息参见[AutoDL Finetuner超参优化功能教程](../../docs/tutorial/autofinetune.md)。
