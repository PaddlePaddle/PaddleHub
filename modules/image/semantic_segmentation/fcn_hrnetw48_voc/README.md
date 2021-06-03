# PaddleHub 图像分割

## 模型预测


若想使用我们提供的预训练模型进行预测，可使用如下脚本：

```python
import paddle
import cv2
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='fcn_hrnetw48_voc')
    img = cv2.imread("/PATH/TO/IMAGE")
    model.predict(images=[img], visualization=True)
```


## 如何开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，通过执行`python train.py`即可开始使用fcn_hrnetw48_voc模型对OpticDiscSeg等数据集进行Fine-tune。

## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

### Step1: 定义数据预处理方式
```python
from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize

transform = Compose([Resize(target_size=(512, 512)), Normalize()])
```

`segmentation_transforms` 数据增强模块定义了丰富的针对图像分割数据的预处理方式，用户可按照需求替换自己需要的数据预处理方式。

### Step2: 下载数据集并使用
```python
from paddlehub.datasets import OpticDiscSeg

train_reader = OpticDiscSeg(transform， mode='train')

```
* `transform`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, `val`, 默认为`train`。

数据集的准备代码可以参考 [opticdiscseg.py](../../paddlehub/datasets/opticdiscseg.py)。`hub.datasets.OpticDiscSeg()`会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。

### Step3: 加载预训练模型

```python
model = hub.Module(name='fcn_hrnetw48_voc', num_classes=2, pretrained=None)
```
* `name`: 选择预训练模型的名字。
* `num_classes`: 分割模型的类别数目。
* `pretrained`: 是否加载自己训练的模型，若为None，则加载提供的模型默认参数。

### Step4: 选择优化策略和运行配置

```python
scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=1000, power=0.9,  end_lr=0.0001)
optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_ocr', use_gpu=True)
```

#### 优化策略

Paddle2.0提供了多种优化器选择，如`SGD`, `Adam`, `Adamax`等，其中`Adam`:

* `learning_rate`: 全局学习率。
*  `parameters`: 待优化模型参数。

#### 运行配置
`Trainer` 主要控制Fine-tune的训练，包含以下可控制的参数:

* `model`: 被优化模型；
* `optimizer`: 优化器选择；
* `use_gpu`: 是否使用gpu，默认为False;
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

我们使用该模型来进行预测。predict.py脚本如下：

```python
import paddle
import cv2
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='fcn_hrnetw48_voc', pretrained='/PATH/TO/CHECKPOINT')
    img = cv2.imread("/PATH/TO/IMAGE")
    model.predict(images=[img], visualization=True)
```

参数配置正确后，请执行脚本`python predict.py`。
**Args**
* `images`:原始图像路径或BGR格式图片；
* `visualization`: 是否可视化，默认为True；
* `save_path`: 保存结果的路径，默认保存路径为'seg_result'。

**NOTE:** 进行预测时，所选择的module，checkpoint_dir，dataset必须和Fine-tune所用的一样。

## 服务部署

PaddleHub Serving可以部署一个在线图像分割服务。

### Step1: 启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m fcn_hrnetw48_voc
```

这样就完成了一个图像分割服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

### Step2: 发送预测请求

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
data = {'images':[cv2_to_base64(org_im)]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/fcn_hrnetw48_voc"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
mask = base64_to_cv2(r.json()["results"][0])
```

### 查看代码

https://github.com/PaddlePaddle/PaddleSeg

### 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.0.0
