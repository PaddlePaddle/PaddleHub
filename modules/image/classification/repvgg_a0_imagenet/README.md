```shell
$ hub install repvgg_a0_imagenet==1.0.0
```

## 命令行预测

```shell
$ hub run repvgg_a0_imagenet --input_path "/PATH/TO/IMAGE" --top_k 5
```

## 脚本预测

```python
import paddle
import paddlehub as hub

if __name__ == '__main__':

    model = hub.Module(name='repvgg_a0_imagenet',)
    result = model.predict([PATH/TO/IMAGE])
```

## Fine-tune代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

### Step1: 定义数据预处理方式
```python
import paddlehub.vision.transforms as T

transforms = T.Compose([T.Resize((256, 256)),
                        T.CenterCrop(224),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])],
                        to_rgb=True)
```

'transforms' 数据增强模块定义了丰富的数据预处理方式，用户可按照需求替换自己需要的数据预处理方式。

### Step2: 下载数据集并使用
```python
from paddlehub.datasets import Flowers

flowers = Flowers(transforms)
flowers_validate = Flowers(transforms, mode='val')
```
* transforms(Callable): 数据预处理方式。
* mode(str): 选择数据模式，可选项有 'train', 'test', 'val'， 默认为'train'。

'hub.datasets.Flowers()' 会自动从网络下载数据集并解压到用户目录下'$HOME/.paddlehub/dataset'目录。


### Step3: 加载预训练模型

```python
import paddlehub as hub

model = hub.Module(name='repvgg_a0_imagenet',
                   label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"],
                   load_checkpoint=None)
```
* name(str): 选择预训练模型的名字。
* label_list(list): 设置标签对应分类类别, 默认为Imagenet2012类别。
* load _checkpoint(str): 模型参数地址。

PaddleHub提供许多图像分类预训练模型，如xception、mobilenet、efficientnet等，详细信息参见[图像分类模型](https://www.paddlepaddle.org.cn/hub?filter=en_category&value=ImageClassification)。

如果想尝试efficientnet模型，只需要更换Module中的'name'参数即可.
```python
import paddlehub as hub

# 更换name参数即可无缝切换efficientnet模型, 代码示例如下
module = hub.Module(name="efficientnetb7_imagenet")
```
**NOTE:**目前部分模型还没有完全升级到2.0版本，敬请期待。

### Step4: 选择优化策略和运行配置

```python
import paddle
from paddlehub.finetune.trainer import Trainer

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt')

trainer.train(flowers, epochs=100, batch_size=32, eval_dataset=flowers_validate, save_interval=1)
```

#### 优化策略

Paddle2.0rc提供了多种优化器选择，如'SGD', 'Adam', 'Adamax'等，详细参见[策略](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/optimizer/optimizer/Optimizer_cn.html)。

其中'Adam':

* learning_rate: 全局学习率。默认为1e-3；
* parameters: 待优化模型参数。

#### 运行配置
'Trainer' 主要控制Fine-tune的训练，包含以下可控制的参数:

* model: 被优化模型；
* optimizer: 优化器选择；
* use_vdl: 是否使用vdl可视化训练过程；
* checkpoint_dir: 保存模型参数的地址；
* compare_metrics: 保存最优模型的衡量指标；

'trainer.train' 主要控制具体的训练过程，包含以下可控制的参数：

* train_dataset: 训练时所用的数据集；
* epochs: 训练轮数；
* batch_size: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* num_workers: works的数量，默认为0；
* eval_dataset: 验证集；
* log_interval: 打印日志的间隔， 单位为执行批训练的次数。
* save_interval: 保存模型的间隔频次，单位为执行训练的轮数。

## 模型预测

当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在'${CHECKPOINT_DIR}/best_model'目录下，其中'${CHECKPOINT_DIR}'目录为Fine-tune时所选择的保存checkpoint的目录。
我们使用该模型来进行预测。predict.py脚本如下：

```python
import paddle
import paddlehub as hub

if __name__ == '__main__':

    model = hub.Module(name='repvgg_a0_imagenet', label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"], load_checkpoint='/PATH/TO/CHECKPOINT')
    result = model.predict(['flower.jpg'])
```

参数配置正确后，请执行脚本'python predict.py'， 加载模型具体可参见[加载](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/framework/io/load_cn.html#load)。

**NOTE:** 进行预测时，所选择的module，checkpoint_dir，dataset必须和Fine-tune所用的一样。

## 服务部署

PaddleHub Serving可以部署一个在线分类任务服务

## Step1: 启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m repvgg_a0_imagenet
```

这样就完成了一个分类任务服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## Step2: 发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64

import numpy as np


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

# 发送HTTP请求
org_im = cv2.imread('/PATH/TO/IMAGE')

data = {'images':[cv2_to_base64(org_im)], 'top_k':2}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/repvgg_a0_imagenet"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
data =r.json()["results"]['data']
```

### 查看代码

[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

### 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.0.0
