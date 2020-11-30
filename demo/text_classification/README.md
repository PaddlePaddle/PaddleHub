# PaddleHub Transformer模型fine-tune文本分类（动态图）

本示例将展示如何使用PaddleHub Transformer模型（如 ERNIE、BERT、RoBERTa等模型）module 以动态图方式fine-tune并完成预测任务。

## 如何开始Fine-tune


我们以中文情感分类公开数据集ChnSentiCorp为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证。通过如下命令，即可启动训练。

```shell
# 设置使用的GPU卡号
export CUDA_VISIBLE_DEVICES=0
python train.py
```


## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

### Step1: 选择模型
```python
import paddlehub as hub

model = hub.Module(name='ernie_tiny', version='2.0.0', task='sequence_classification')
```

其中，参数：

* `name`：模型名称，可以选择`ernie`，`ernie-tiny`，`bert_chinese_L-12_H-768_A-12`，`chinese-roberta-wwm-ext`，`chinese-roberta-wwm-ext-large`等。
* `version`：module版本号
* `task`：fine-tune任务。此处为`sequence_classification`，表示文本分类任务。

### Step2: 下载并加载数据集

```python
train_dataset = hub.datasets.ChnSentiCorp(
    tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='train')
dev_dataset = hub.datasets.ChnSentiCorp(
    tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='dev')
```

* `tokenizer`：表示该module所需用到的tokenizer，其将对输入文本完成切词，并转化成module运行所需模型输入格式。
* `mode`：选择数据模式，可选项有 `train`, `test`, `val`， 默认为`train`。
* `max_seq_len`：ERNIE/BERT模型使用的最大序列长度，若出现显存不足，请适当调低这一参数。

### Step3:  选择优化策略和运行配置

```python
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls')

trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset)

# 在测试集上评估当前训练模型
trainer.evaluate(test_dataset, batch_size=32)
```

#### 优化策略

Paddle2.0-rc提供了多种优化器选择，如`SGD`, `Adam`, `Adamax`等，详细参见[策略](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/optimizer/optimizer/Optimizer_cn.html)。

其中`Adam`:

* `learning_rate`: 全局学习率。默认为1e-3；
* `parameters`: 待优化模型参数。

#### 运行配置

`Trainer` 主要控制Fine-tune的训练，包含以下可控制的参数:

* `model`: 被优化模型；
* `optimizer`: 优化器选择；
* `use_vdl`: 是否使用vdl可视化训练过程；
* `checkpoint_dir`: 保存模型参数的地址；
* `compare_metrics`: 保存最优模型的衡量指标；

`trainer.train` 主要控制具体的训练过程，包含以下可控制的参数：

* `train_dataset`: 训练时所用的数据集；
* `epochs`: 训练轮数；
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* `num_workers`: works的数量，默认为0；
* `eval_dataset`: 验证集；
* `log_interval`: 打印日志的间隔， 单位为执行批训练的次数。
* `save_interval`: 保存模型的间隔频次，单位为执行训练的轮数。

## 模型预测

当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为Fine-tune时所选择的保存checkpoint的目录。

我们以以下数据为待预测数据，使用该模型来进行预测

```text
这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般
怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片
作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。
```

```python
import paddlehub as hub

data = [
    ['这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'],
    ['怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'],
    ['作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'],
]
label_map = {0: 'negative', 1: 'positive'}

model = hub.Module(
    directory='/mnt/zhangxuefei/program-paddle/PaddleHub/modules/text/language_model/ernie_tiny',
    version='2.0.0',
    task='sequence_classification',
    load_checkpoint='./test_ernie_text_cls/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text[0], results[idx]))
```

参数配置正确后，请执行脚本`python predict.py`， 加载模型具体可参见[加载](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/framework/io/load_cn.html#load)。

### 依赖

paddlepaddle >= 2.0.0rc

paddlehub >= 2.0.0
