# 自定义数据

训练一个新任务时，如果从零开始训练时，这将是一个耗时的过程，并且效果可能达不到理想的效果，此时您可以利用PaddleHub提供的预训练模型进行具体任务的Fine-tune。您只需要对自定义数据进行相应的预处理，随后输入预训练模型中，即可得到相应的结果。请参考如下内容设置数据集的结构。

## 一、图像分类数据集

利用PaddleHub迁移分类任务使用自定义数据时，需要切分数据集，将数据集切分为训练集、验证集和测试集。

### 数据准备

需要三个文本文件来记录对应的图片路径和标签，此外还需要一个标签文件用于记录标签的名称。
```
├─data: 数据目录
  ├─train_list.txt：训练集数据列表
  ├─test_list.txt：测试集数据列表
  ├─validate_list.txt：验证集数据列表
  ├─label_list.txt：标签列表
  └─...
```
训练/验证/测试集的数据列表文件的格式如下
```
图片1路径 图片1标签
图片2路径 图片2标签
...
```
label_list.txt的格式如下
```
分类1名称
分类2名称
...
```

示例：
以[Flower数据集](../reference/datasets.md)为示例，train_list.txt/test_list.txt/validate_list.txt内容如下示例
```
roses/8050213579_48e1e7109f.jpg 0
sunflowers/45045003_30bbd0a142_m.jpg 3
daisy/3415180846_d7b5cced14_m.jpg 2
```

label_list.txt内容如下：
```
roses
tulips
daisy
sunflowers
dandelion
```

### 数据集加载

数据集的准备代码可以参考 [flowers.py](../../paddlehub/datasets/flowers.py)。`hub.datasets.Flowers()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。具体使用如下：

```python
from paddlehub.datasets import Flowers

flowers = Flowers(transforms)
flowers_validate = Flowers(transforms, mode='val')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, `val`， 默认为`train`。

## 二、图像着色数据集

利用PaddleHub迁移着色任务使用自定义数据时，需要切分数据集，将数据集切分为训练集和测试集。

### 数据准备

需要将准备用于着色训练和测试的彩色图像分成训练集数据和测试集数据。
```
├─data: 数据目录
  ├─train：训练集数据
      |-图片文件夹1
      |-图片文件夹2
      |-……
      |-图片1
      |-图片2
      |-……

  ├─test：测试集数据
    |-图片文件夹1
    |-图片文件夹2
    |-……
    |-图片1
    |-图片2
    |-……
  └─……
```

示例：
PaddleHub为用户提供了用于着色的数据集`Canvas数据集`， 它由1193张莫奈风格和400张梵高风格的图像组成，以[Canvas数据集](../reference/datasets.md)为示例，train文件夹内容如下:

```
├─train：训练集数据
      |-monet
          |-图片1
          |-图片2
          |-……  
      |-vango
          |-图片1
          |-图片2
          |-……
```

### 数据集加载

数据集的准备代码可以参考 [canvas.py](../../paddlehub/datasets/canvas.py)。`hub.datasets.Canvas()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。具体使用如下：

```python
from paddlehub.datasets import Canvas

color_set = Canvas(transforms, mode='train')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, 默认为`train`。

## 三、风格迁移数据集

利用PaddleHub进行风格迁移任务使用自定义数据时，需要切分数据集，将数据集切分为训练集和测试集。

### 数据准备

需要将准备用于风格迁移的彩色图像分成训练集和测试集数据。

```
├─data: 数据目录
  ├─train：训练集数据
      |-图片文件夹1
      |-图片文件夹2
      |-...
      |-图片1
      |-图片2
      |-...

  ├─test：测试集数据
    |-图片文件夹1
    |-图片文件夹2
    |-...
    |-图片1
    |-图片2
    |-...
  |- 21styles
    ｜-图片1
    ｜-图片2
  └─...
```

示例：
PaddleHub为用户提供了用于风格迁移的数据集`MiniCOCO数据集`, 训练集数据和测试集数据来源于COCO2014， 其中训练集有2001张图片，测试集有200张图片。 `21styles`文件夹下存放着21张不同风格的图片，用户可以根据自己的需求更换不同风格的图片。以[MiniCOCO数据集](../reference/datasets.md)为示例，train文件夹内容如下:

```
├─train：训练集数据
      |-train
          |-图片1
          |-图片2
          |-……  
      |-test
          |-图片1
          |-图片2
          |-……
      |-21styles
          |-图片1
          |-图片2
          |-……
```

### 数据集加载

数据集的准备代码可以参考 [minicoco.py](../../paddlehub/datasets/minicoco.py)。`hub.datasets.MiniCOCO()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。具体使用如下：

```python
from paddlehub.datasets import MiniCOCO

ccolor_set = MiniCOCO(transforms, mode='train')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, 默认为`train`。
