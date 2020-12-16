# PaddleHub Transformer模型fine-tune序列标注（动态图）

本示例将展示如何使用PaddleHub Transformer模型（如 ERNIE、BERT、RoBERTa等模型）Module 以动态图方式fine-tune并完成预测任务。

## 如何开始Fine-tune


我们以微软亚洲研究院发布的中文实体识别数据集MSRA-NER为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证。通过如下命令，即可启动训练。

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

model = hub.Module(name='ernie_tiny', version='2.0.1', task='token-cls')
```

其中，参数：

* `name`：模型名称，可以选择`ernie`，`ernie-tiny`，`bert_chinese_L-12_H-768_A-12`，`chinese-roberta-wwm-ext`，`chinese-roberta-wwm-ext-large`等。
* `version`：module版本号
* `task`：fine-tune任务。此处为`token-cls`，表示序列标注任务。

### Step2: 下载并加载数据集

```python
train_dataset = hub.datasets.MSRA_NER(
    tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=50, mode='train')
dev_dataset = hub.datasets.MSRA_NER(
    tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=50, mode='dev')
```

* `tokenizer`：表示该module所需用到的tokenizer，其将对输入文本完成切词，并转化成module运行所需模型输入格式。
* `mode`：选择数据模式，可选项有 `train`, `test`, `val`， 默认为`train`。
* `max_seq_len`：ERNIE/BERT模型使用的最大序列长度，若出现显存不足，请适当调低这一参数。

### Step3:  选择优化策略和运行配置

```python
optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_token_cls', use_gpu=False)

trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset)

# 在测试集上评估当前训练模型
trainer.evaluate(test_dataset, batch_size=32)
```

#### 优化策略

Paddle2.0-rc提供了多种优化器选择，如`SGD`, `Adam`, `Adamax`, `AdamW`等，详细参见[策略](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/optimizer/optimizer/Optimizer_cn.html)。

其中`AdamW`:

* `learning_rate`: 全局学习率。默认为1e-3；
* `parameters`: 待优化模型参数。

#### 运行配置

`Trainer` 主要控制Fine-tune的训练，包含以下可控制的参数:

* `model`: 被优化模型；
* `optimizer`: 优化器选择；
* `use_gpu`: 是否使用GPU训练，默认为False;
* `use_vdl`: 是否使用vdl可视化训练过程；
* `checkpoint_dir`: 保存模型参数的地址；
* `compare_metrics`: 保存最优模型的衡量指标；

`trainer.train` 主要控制具体的训练过程，包含以下可控制的参数：

* `train_dataset`: 训练时所用的数据集；
* `epochs`: 训练轮数；
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* `num_workers`: workers的数量，默认为0；
* `eval_dataset`: 验证集；
* `log_interval`: 打印日志的间隔， 单位为执行批训练的次数。
* `save_interval`: 保存模型的间隔频次，单位为执行训练的轮数。

## 模型预测

当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为Fine-tune时所选择的保存checkpoint的目录。

我们以以下数据为待预测数据，使用该模型来进行预测

```text
去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出三问：『假如上访群众是我们的父母姐妹，你会用什么样的感情对待他们？
新华社北京5月7日电国务院副总理李岚清今天在中南海会见了美国前商务部长芭芭拉·弗兰克林。
根据测算，海卫1表面温度已经从“旅行者”号探测器1989年造访时的零下236摄氏度上升到零下234摄氏度。
华裔作家韩素音女士曾三次到大足，称“大足石窟是一座未被开发的金矿”。
```

```python
import paddlehub as hub

split_char = "\002"
label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
text_a = [
    '去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出三问：『假如上访群众是我们的父母姐妹，你会用什么样的感情对待他们？',
    '新华社北京5月7日电国务院副总理李岚清今天在中南海会见了美国前商务部长芭芭拉·弗兰克林。',
    '根据测算，海卫1表面温度已经从“旅行者”号探测器1989年造访时的零下236摄氏度上升到零下234摄氏度。',
    '华裔作家韩素音女士曾三次到大足，称“大足石窟是一座未被开发的金矿”。',
]
data = [[split_char.join(text)] for text in text_a]
label_map = {
    idx: label for idx, label in enumerate(label_list)
}

model = hub.Module(
    name='ernie_tiny',
    version='2.0.1',
    task='token_cls',
    load_checkpoint='./token_cls_save_dir/best_model/model.pdparams',
    label_map=label_map,
)

results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)
for idx, text in enumerate(text_a):
    print(f'Data: {text} \t Lable: {", ".join(results[idx][1:len(text)+1])}')
```

参数配置正确后，请执行脚本`python predict.py`， 加载模型具体可参见[加载](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/framework/io/load_cn.html#load)。

### 依赖

paddlepaddle >= 2.0.0rc

paddlenlp >= 2.0.0a4

paddlehub >= 2.0.0
