# 声音分类

本示例展示如何使用PaddleHub Fine-tune API以及CNN14等预训练模型完成声音分类和Tagging的任务。

CNN14等预训练模型的详情，请参考论文[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf)和代码[audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)。


## 如何开始Fine-tune

我们以环境声音分类公开数据集[ESC50](https://github.com/karolpiczak/ESC-50)为示例数据集，可以运行下面的命令，在训练集（train.npz）上进行模型训练，并在开发集（dev.npz）验证。通过如下命令，即可启动训练。

```python
# 设置使用的GPU卡号
export CUDA_VISIBLE_DEVICES=0
python train.py
```


## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

### Step1: 选择模型

```python
import paddle
import paddlehub as hub
from paddlehub.datasets import ESC50

model = hub.Module(name='panns_cnn14', version='1.0.0', task='sound-cls', num_class=ESC50.num_class)
```

其中，参数：
- `name`: 模型名称，可以选择`panns_cnn14`、`panns_cnn10` 和`panns_cnn6`，具体的模型参数信息可见下表。
- `version`: module版本号
- `task`：模型的执行任务。`sound-cls`表示声音分类任务；`None`表示Audio Tagging任务。
- `num_classes`：表示当前声音分类任务的类别数，根据具体使用的数据集确定。

目前可选用的预训练模型：
模型名      | PaddleHub Module
-----------| :------:
CNN14      | `hub.Module(name='panns_cnn14')`
CNN10      | `hub.Module(name='panns_cnn10')`
CNN6       | `hub.Module(name='panns_cnn6')`

### Step2: 加载数据集

```python
train_dataset = ESC50(mode='train')
dev_dataset = ESC50(mode='dev')
```

### Step3: 选择优化策略和运行配置

```python
optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./', use_gpu=True)
```

#### 优化策略

Paddle2.0提供了多种优化器选择，如`SGD`, `AdamW`, `Adamax`等，详细参见[策略](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)。

其中`AdamW`:

- `learning_rate`: 全局学习率。默认为1e-3；
- `parameters`: 待优化模型参数。

其余可配置参数请参考[AdamW](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/adamw/AdamW_cn.html#cn-api-paddle-optimizer-adamw)。

#### 运行配置

`Trainer` 主要控制Fine-tune的训练，包含以下可控制的参数:

- `model`: 被优化模型；
- `optimizer`: 优化器选择；
- `use_vdl`: 是否使用vdl可视化训练过程；
- `checkpoint_dir`: 保存模型参数的地址；
- `compare_metrics`: 保存最优模型的衡量指标；


### Step4: 执行训练和模型评估

```python
trainer.train(
    train_dataset,
    epochs=50,
    batch_size=16,
    eval_dataset=dev_dataset,
    save_interval=10,
)
trainer.evaluate(dev_dataset, batch_size=16)
```

`trainer.train`执行模型的训练，其参数可以控制具体的训练过程，主要的参数包含：

- `train_dataset`: 训练时所用的数据集；
- `epochs`: 训练轮数；
- `batch_size`: 训练时每一步用到的样本数目，如果使用GPU，请根据实际情况调整batch_size；
- `num_workers`: works的数量，默认为0；
- `eval_dataset`: 验证集；
- `log_interval`: 打印日志的间隔， 单位为执行批训练的次数。
- `save_interval`: 保存模型的间隔频次，单位为执行训练的轮数。

`trainer.evaluate`执行模型的评估，主要的参数包含：

- `eval_dataset`: 模型评估时所用的数据集；
- `batch_size`: 模型评估时每一步用到的样本数目，如果使用GPU，请根据实际情况调整batch_size


## 模型预测

当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为Fine-tune时所选择的保存checkpoint的目录。

以下代码将本地的音频文件`./cat.wav`作为预测数据，使用训好的模型对它进行分类，输出结果。

```python
import os

import librosa

import paddlehub as hub
from paddlehub.datasets import ESC50

wav = './cat.wav'  # 存储在本地的需要预测的wav文件
sr = 44100  # 音频文件的采样率
checkpoint = './best_model/model.pdparams'  # 模型checkpoint

label_map = {idx: label for idx, label in enumerate(ESC50.label_list)}

model = hub.Module(name='panns_cnn14',
                    version='1.0.0',
                    task='sound-cls',
                    num_class=ESC50.num_class,
                    label_map=label_map,
                    load_checkpoint=checkpoint)

data = [librosa.load(wav, sr=sr)[0]]
result = model.predict(data, sample_rate=sr, batch_size=1, feat_type='mel', use_gpu=True)

print(result[0])  # result[0]包含音频文件属于各类别的概率值
```


## Audio Tagging

当前使用的模型是基于[Audioset数据集](https://research.google.com/audioset/)的预训练模型，除了以上的针对特定声音分类数据集的finetune任务，模型还支持基于Audioset 527个标签的Tagging功能。

以下代码将本地的音频文件`./cat.wav`作为预测数据，使用预训练模型对它进行打分，输出top 10的标签和对应的得分。

```python
import os

import librosa
import numpy as np

import paddlehub as hub
from paddlehub.env import MODULE_HOME


wav = './cat.wav'  # 存储在本地的需要预测的wav文件
sr = 44100  # 音频文件的采样率
topk = 10  # 展示音频得分前10的标签和分数

# 读取audioset数据集的label文件
label_file = os.path.join(MODULE_HOME, 'panns_cnn14', 'audioset_labels.txt')
label_map = {}
with open(label_file, 'r') as f:
    for i, l in enumerate(f.readlines()):
        label_map[i] = l.strip()

model = hub.Module(name='panns_cnn14', version='1.0.0', task=None, label_map=label_map)

data = [librosa.load(wav, sr=sr)[0]]
result = model.predict(data, sample_rate=sr, batch_size=1, feat_type='mel', use_gpu=True)

# 打印topk的类别和对应得分
msg = ''
for label, score in list(result[0].items())[:topk]:
    msg += f'{label}: {score}\n'
print(msg)
```

### 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.1.0
