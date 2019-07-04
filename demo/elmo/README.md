# PaddleHub 文本分类

本示例将展示如何使用PaddleHub Finetune API以及加载ELMo预训练中文word embedding在中文情感分析数据集ChnSentiCorp上完成分类任务。

## 如何开始Finetune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_elmo_finetune.sh`即可开始使用ELMo对ChnSentiCorp数据集进行Finetune。

其中脚本参数说明如下：

```bash
# 模型相关
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数use
--use_gpu: 是否使用GPU进行FineTune，默认为True
--learning_rate: Finetune的最大学习率
--weight_decay: 控制正则项力度的参数，用于防止过拟合，默认为0.01
--warmup_proportion: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0
--num_epoch: Finetune迭代的轮数


# 任务相关
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型
```

## 代码步骤

使用PaddleHub Finetune API进行Finetune可以分为4个步骤

### Step1: 加载预训练模型

```python
module = hub.Module(name="elmo")
inputs, outputs, program = module.context(trainable=True)
```

### Step2: 准备数据集并使用LACClassifyReader读取数据
```python
dataset = hub.dataset.ChnSentiCorp()
reader = hub.reader.LACClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path())
```

其中数据集的准备代码可以参考 [chnsenticorp.py](https://github.com/PaddlePaddle/PaddleHub/blob/develop/paddlehub/dataset/chnsenticorp.py)

`hub.dataset.ChnSentiCorp()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录

`module.get_vaocab_path()` 会返回预训练模型对应的词表

LACClassifyReader中的`data_generator`会自动按照模型对应词表对数据进行切词，以迭代器的方式返回ELMo所需要的Tensor格式，包括`word_ids`.

### Step3：选择优化策略和运行配置

```python
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_proportion=0.0,
    lr_scheduler="linear_decay",
)

config = hub.RunConfig(use_cuda=True, use_data_parallel=True, use_pyreader=False, num_epoch=3, batch_size=32, strategy=strategy)
```

#### 优化策略
针对ERNIE与BERT类任务，PaddleHub封装了适合这一任务的迁移学习优化策略`AdamWeightDecayStrategy`

* `learning_rate`: Finetune过程中的最大学习率;
* `weight_decay`: 模型的正则项参数，默认0.01，如果模型有过拟合倾向，可适当调高这一参数;
* `warmup_proportion`: 如果warmup_proportion>0, 例如0.1, 则学习率会在前10%的steps中线性增长至最高值learning_rate;
* `lr_scheduler`: 有两种策略可选(1) `linear_decay`策略学习率会在最高点后以线性方式衰减; `noam_decay`策略学习率会在最高点以多项式形式衰减；

#### 运行配置
`RunConfig` 主要控制Finetune的训练，包含以下可控制的参数:

* `log_interval`: 进度日志打印间隔，默认每10个step打印一次
* `eval_interval`: 模型评估的间隔，默认每100个step评估一次验证集
* `save_ckpt_interval`: 模型保存间隔，请根据任务大小配置，默认只保存验证集效果最好的模型和训练结束的模型
* `use_cuda`: 是否使用GPU训练，默认为False
* `use_data_parallel`: 是否使用并行计算，默认False。打开该功能依赖nccl库
* `use_pyreader`: 是否使用pyreader，默认False
* `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成
* `num_epoch`: finetune的轮数
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size
* `enable_memory_optim`: 是否使用内存优化， 默认为True
* `strategy`: Finetune优化策略

**Note**: 当使用LACClassifyReader时，use_pyreader应该为False。

### Step4: 构建网络并创建分类迁移任务进行Finetune

有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。
>* 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
>* 从输出变量中找到输入单词对应的elmo_embedding, 并拼接上随机初始化word embedding；
>* 在拼接embedding输入gru网络，进行文本分类，生成Task；

```python
word_ids = inputs["word_ids"]
elmo_embedding = outputs["elmo_embed"]

feed_list = [word_ids.name]

switch_main_program(program)

word_embed_dims = 128
word_embedding = fluid.layers.embedding(
    input=word_ids,
    size=[word_dict_len, word_embed_dims],
    param_attr=fluid.ParamAttr(
        learning_rate=30,
        initializer=fluid.initializer.Uniform(low=-0.1, high=0.1)))

input_feature = fluid.layers.concat(
    input=[elmo_embedding, word_embedding], axis=1)

fc = gru_net(program, input_feature)

elmo_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=fc,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config)

elmo_task.finetune_and_eval()
```
**NOTE:**
1. `outputs["elmo_embed"]`返回了ELMo模型预训练的word embedding。
2. `hub.TextClassifierTask`通过输入特征，label与迁移的类别数，可以生成适用于文本分类的迁移任务`TextClassifierTask`

## VisualDL 可视化

Finetune API训练过程中会自动对关键训练指标进行打点，启动程序后执行下面命令
```bash
$ visualdl --logdir $CKPT_DIR/vdllog -t ${HOST_IP}
```
其中${HOST_IP}为本机IP地址，如本机IP地址为192.168.0.1，用浏览器打开192.168.0.1:8040，其中8040为端口号，即可看到训练过程中指标的变化情况

## 模型预测

通过Finetune完成模型训练后，在对应的ckpt目录下，会自动保存验证集上效果最好的模型。
配置脚本参数
```
CKPT_DIR="./ckpt_chnsentiment"
python predict.py --checkpoint_dir --use_gpu True
```
其中CKPT_DIR为Finetune API保存最佳模型的路径

参数配置正确后，请执行脚本`sh run_predict.sh`，即可看到以下文本分类预测结果, 以及最终准确率。
如需了解更多预测步骤，请参考`predict.py`
