# PaddleHub 图像分类

本示例将展示如何使用PaddleHub对预训练模型进行finetune并完成预测任务。

## 如何开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，通过执行`python train.py`即可开始使用mobilenet_v2_imagenet对[Flowers](../../docs/reference/dataset.md#class-hubdatasetsflowers)等数据集进行Fine-tune。

## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

### Step1: 定义数据预处理方式
```python
import paddlehub.vision.transforms as T

transforms = T.Compose([T.Resize((224, 224)), T.Normalize()])
```

`transforms` 数据增强模块定义了丰富的数据预处理方式，用户可按照需求替换自己需要的数据预处理方式。

### Step2: 下载数据集并使用
```python
from paddlehub.datasets import Flowers

flowers = Flowers(transforms)

flowers_validate = Flowers(transforms, mode='val')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, `val`， 默认为`train`。

数据集的准备代码可以参考 [flowers.py](../../paddlehub/datasets/flowers.py)。`hub.datasets.Flowers()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。


### Step3: 加载预训练模型

```python
module = hub.Module(name="resnet50_vd_imagenet_ssld", class_dim=flowers.num_classes)
```
* `name`: 选择预训练模型的名字。
* `class_dim`: 设置最终输出分类类别。

PaddleHub提供许多图像分类预训练模型，如xception、mobilenet、efficientnet等，详细信息参见[图像分类模型](https://www.paddlepaddle.org.cn/hub?filter=en_category&value=ImageClassification)。
目前部分模型还没有完全升级到2.0版本，敬请期待。

如果想尝试efficientnet模型，只需要更换Module中的`name`参数即可.
```python
# 更换name参数即可无缝切换efficientnet模型, 代码示例如下
module = hub.Module(name="efficientnetb7_imagenet")
```

### Step4: 选择优化策略和运行配置

```python
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt')

trainer.train(flowers, epochs=100, batch_size=32, eval_dataset=flowers_validate, save_interval=1)
```

#### 优化策略

Paddle2.0-rc提供了多种优化器选择，如`SGD`, `Adam`, `Adamax`等，详细参见[策略](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/optimizer/optimizer/Optimizer_cn.html)。

其中`Adam`:

* `learning_rate`: 全局学习率。默认为1e-3；
*  `parameters`: 待优化模型参数。

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

我们使用该模型来进行预测。predict.py脚本如下：

```python
import paddle
import paddlehub as hub

if __name__ == '__main__':

    model = hub.Module(name='mobilenet_v2_imagenet', class_dim=5, load_checkpoint=/PATH/TO/CHECKPOINT)
    result = model.predict('flower.jpg')
```

参数配置正确后，请执行脚本`python predict.py`， 加载模型具体可参见[加载](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/framework/io/load_cn.html#load)。

**NOTE:** 进行预测时，所选择的module，checkpoint_dir，dataset必须和Fine-tune所用的一样。

## 超参优化AutoDL Finetuner

PaddleHub还提供了超参优化（Hyperparameter Tuning）功能， 自动搜索最优模型超参得到更好的模型效果。详细信息参见[AutoDL Finetuner超参优化功能教程](../../docs/tutorial/autofinetune.md)。
