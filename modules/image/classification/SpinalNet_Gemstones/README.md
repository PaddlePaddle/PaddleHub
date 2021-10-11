# PaddleHub SpinalNet

本示例将展示如何使用PaddleHub的SpinalNet预训练模型进行宝石识别或finetune并完成宝石的预测任务。

## 1. 首先要安装PaddleHub2.0版

```shell
$pip install -U paddlehub==2.0.0
```

## 2. 在本地加载封装的模型

```Python
import paddlehub as hub
```
### 加载spinalnet_res50_gemstone
```Python
spinal_res50 = hub.Module(name="spinalnet_res50_gemstone")
```
### 加载spinalnet_vgg16_gemstone
```Python
spinal_vgg16 = hub.Module(name="spinalnet_vgg16_gemstone")
```
### 加载spinalnet_res101_gemstone
```Python
spinal_res101 = hub.Module(name="spinalnet_res101_gemstone")
```
## 3. 预测

### 使用spinalnet_res50_gemstone预测
```Python
result_res50 = spinal_res50.predict(['/PATH/TO/IMAGE'])
print(result_res50)
```
### 使用spinalnet_vgg16_gemstone预测
```Python
result_vgg16 = spinal_vgg16.predict(['/PATH/TO/IMAGE'])
print(result_vgg16)
```
### 使用spinalnet_res101_gemstone预测
```Python
sresult_res101 = spinal_res101.predict(['/PATH/TO/IMAGE'])
print(result_res101)
```
## 4. 命令行预测

```shell
$ hub run spinalnet_res50_gemstone --input_path "/PATH/TO/IMAGE" --top_k 5
```

## 5. 对PaddleHub模型进行训练微调

## 如何开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，即可对Spinalnet模型进行针对宝石数据集的Fine-tune。

## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为5个步骤。

### Step1: 加载必要的库
```python
from paddlehub.finetune.trainer import Trainer
from gem_dataset import GemStones
from paddlehub.vision import transforms as T
import paddle
```


### Step2: 定义数据预处理方式
```python

train_transforms = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])], to_rgb=True)
eval_transforms = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])], to_rgb=True)
```

`transforms` 数据增强模块定义了丰富的数据预处理方式，用户可按照需求替换自己需要的数据预处理方式。

### Step3: 定义数据集
```python
gem_train = GemStones(transforms=train_transforms, mode='train')
gem_validate = GemStones(transforms=eval_transforms, mode='eval')
```


数据集的准备代码可以参考 [gem_dataset.py](PaddleHub/modules/thirdparty/image/classification/SpinanlNet_Gemstones/gem_dataset.py)。


### Step4: 开始训练微调

```python
optimizer = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=spinal_res50.parameters())
trainer = Trainer(spinal_res50, optimizer, use_gpu=True, checkpoint_dir='fine_tuned_model')
trainer.train(gem_train, epochs=5, batch_size=128, eval_dataset=gem_validate, save_interval=1, log_interval=10)
```

### Step5: 微调后再预测

```python
spinal_res50 = hub.Module(name="spinalnet_res50_gemstone")
result_res50 = spinal_res50.predict(['/PATH/TO/IMAGE'])
print(result_res50)
```


### 查看代码

https://github.com/PaddleHub/modules/thirdparty/image/classification/SpinalNet_Gemstones/spinalnet_res50_gemstone/module.py

https://github.com/PaddleHub/modules/thirdparty/image/classification/SpinalNet_Gemstones/spinalnet_res101_gemstone/module.py

https://github.com/PaddleHub/modules/thirdparty/image/classification/SpinalNet_Gemstones/spinalnet_vgg16_gemstone/module.py

### 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.0.0
