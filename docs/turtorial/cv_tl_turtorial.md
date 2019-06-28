# 如何使用PaddleHub进行图像分类迁移

Fine-tune是迁移学习中使用得最多的方式之一。

其主要思想是通过对预训练模型进行结构和参数的 **微调**  来实现模型迁移，从而达到模型适应新领域(Domain)数据的目的。

本文以Kaggle的猫狗分类数据集为例子，详细了介绍如何在PaddleHub中进行CV方向的finetune。

## 一、准备工作

在开始进行fine-tune前，我们需要完成以下工作准备：

### 1. 安装PaddlePaddle

PaddleHub是基于PaddlePaddle的预训练模型管理框架，使用PaddleHub前需要先安装PaddlePaddle，如果您本地已经安装了CPU或者GPU版本的PaddlePaddle，那么可以跳过以下安装步骤。

```shell
# 安装cpu版本的PaddlePaddle
$ pip install paddlepaddle
```

我们推荐您使用大于1.4.0版本的PaddlePaddle，如果您本地版本较低，使用如下命令进行升级
```shell
$ pip install --upgrade paddlepaddle
```

在安装过程中如果遇到问题，您可以到[Paddle官方网站](http://www.paddlepaddle.org/)上查看解决方案

### 2. 安装PaddleHub

通过以下命令来安装PaddleHub

```shell
$ pip install paddlehub
```

如果在安装过程中遇到问题，您可以查看下[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)来查找问题解决方案，如果无法解决，请在[Issue](https://github.com/PaddlePaddle/PaddleHub/issues)中反馈问题，我们会尽快分析解决

## 二、选择合适的模型

首先导入必要的python包

```python
# -*- coding: utf8 -*-
import paddlehub as hub
import paddle.fluid as fluid
```

接下来我们要在PaddleHub中选择合适的预训练模型来Finetune，由于猫狗分类是一个图像分类任务，因此我们使用经典的ResNet-50作为预训练模型。PaddleHub提供了丰富的图像分类预训练模型，包括了最新的神经网络架构搜索类的PNASNet，我们推荐您尝试不同的预训练模型来获得更好的性能。

```python
module_map = {
    "resnet50": "resnet_v2_50_imagenet",
    "resnet101": "resnet_v2_101_imagenet",
    "resnet152": "resnet_v2_152_imagenet",
    "mobilenet": "mobilenet_v2_imagenet",
    "nasnet": "nasnet_imagenet",
    "pnasnet": "pnasnet_imagenet"
}

module_name = module_map["resnet50"]
module = hub.Module(name = module_name)
```

## 三、数据准备

接着需要加载图片数据集。为了快速体验，我们直接加载paddlehub提供的猫狗分类数据集，如果想要使用自定义的数据进行体验，请查看`自定义数据`

```python
# 直接用PaddleHub提供的数据集
dataset = hub.dataset.DogCat()
```

## 四、自定义数据

本节说明如何组装自定义的数据，如果想使用猫狗数据集进行体验，可以直接跳过本节。

使用自定义数据时，我们需要自己切分数据集，将数据集且分为训练集、验证集和测试集。

同时使用三个文本文件来记录对应的图片路径和标签，此外还需要一个标签文件用于记录标签的名称。
```
├─data: 数据目录
  ├─train_list.txt：训练集数据列表
  ├─test_list.txt：测试集数据列表
  ├─validate_list.txt：验证集数据列表
  ├─label_list.txt：标签列表
  └─……
```
训练/验证/测试集的数据列表文件的格式如下
```
图片1路径 图片1标签
图片2路径 图片2标签
...
```
标签列表文件的格式如下
```
分类1名称
分类2名称
...
```
使用如下的方式进行加载数据，生成数据集对象

`注意事项`：
1. num_labels要填写实际的分类数量，如猫狗分类该字段值为2，food101该字段值为101，下文以`2`为例子
2. base_path为数据集实际路径，需要填写全路径，下文以`/test/data`为例子
3. 训练/验证/测试集的数据列表文件中的图片路径需要相对于base_path的相对路径，例如图片的实际位置为`/test/data/dog/dog1.jpg`，base_path为`/test/data`，则文件中填写的路径应该为`dog/dog1.jpg`

```python
# 使用本地数据集
class MyDataSet(hub.dataset.base_cv_dataset.ImageClassificationDataset):
    def __init__(self):
        self.base_path = "/test/data"
        self.train_list_file = "train_list.txt"
        self.test_list_file = "test_list.txt"
        self.validate_list_file = "validate_list.txt"
        self.label_list_file = "label_list.txt"
        self.label_list = None
        self.num_labels = 2

dataset = MyDataSet()
```

## 五、生成Reader

接着生成一个图像分类的reader，reader负责将dataset的数据进行预处理，接着以特定格式组织并输入给模型进行训练。

当我们生成一个图像分类的reader时，需要指定输入图片的大小

```python
data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)
```

## 六、组建Finetune Task

有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。

由于猫狗分类是一个二分类的任务，而我们下载的分类module是在ImageNet数据集上训练的千分类模型，所以我们需要对模型进行简单的微调，把模型改造为一个二分类模型：

1. 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
2. 从输出变量中找到特征图提取层feature_map；
3. 在feature_map后面接入一个全连接层，生成Task；

```python
input_dict, output_dict, program = module.context(trainable=True)

img = input_dict["image"]
feature_map = output_dict["feature_map"]

task = hub.create_img_cls_task(
    feature=feature_map, num_classes=dataset.num_labels)

feed_list = [img.name, task.variable("label").name]
```

## 七、选择运行时配置

在进行Finetune前，我们可以设置一些运行时的配置，例如如下代码中的配置，表示：

> `use_cuda`：设置为False表示使用CPU进行训练。如果您本机支持GPU，且安装的是GPU版本的PaddlePaddle，我们建议您将这个选项设置为True；
>
> `epoch`：要求Finetune的任务只遍历1次训练集；
>
> `batch_size`：每次训练的时候，给模型输入的每批数据大小为32，模型训练时能够并行处理批数据，因此batch_size越大，训练的效率越高，但是同时带来了内存的负荷，过大的batch_size可能导致内存不足而无法训练，因此选择一个合适的batch_size是很重要的一步；
>
> `log_interval`：每隔10 step打印一次训练日志；
>
> `eval_interval`：每隔50 step在验证集上进行一次性能评估；
>
> `checkpoint_dir`：将训练的参数和数据保存到cv_finetune_turtorial_demo目录中；
>
> `strategy`：使用DefaultFinetuneStrategy策略进行finetune；

更多运行配置，请查看[RunConfig](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/api/RunConfig.md)

```python
config = hub.RunConfig(
    use_cuda=False,
    num_epoch=1,
    checkpoint_dir="cv_finetune_turtorial_demo",
    batch_size=32,
    log_interval=10,
    eval_interval=50,
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())
```

## 八、开始Finetune

我们选择`finetune_and_eval`接口来进行模型训练，这个接口在finetune的过程中，会周期性的进行模型效果的评估，以便我们了解整个训练过程的性能变化。

```python
hub.finetune_and_eval(
    task, feed_list=feed_list, data_reader=data_reader, config=config)
```

## 九、查看训练过程的效果

训练过程中的性能数据会被记录到本地，我们可以通过visualdl来可视化这些数据。

我们在shell中输入以下命令来启动visualdl，其中`${HOST_IP}`为本机IP，需要用户自行指定
```shell
$ visualdl --logdir ./cv_finetune_turtorial_demo/vdllog --host ${HOST_IP} --port 8989
```

启动服务后，我们使用浏览器访问`${HOST_IP}:8989`，可以看到训练以及预测的loss曲线和accuracy曲线
![img](https://github.com/PaddlePaddle/PaddleHub/blob/release/v0.5.0/docs/imgs/cv_turtorial_vdl_log.jpg)

## 十、使用模型进行预测

当Finetune完成后，我们使用模型来进行预测，整个预测流程大致可以分为以下几步：
1. 构建网络
2. 生成预测数据的Reader
3. 切换到预测的Program
4. 加载预训练好的参数
5. 运行Program进行预测

通过以下命令来获取测试的图片
`注意`：以下示例仍然以猫狗分类为例子，其他数据集所用的测试图片请自行准备
```shell
$ wget --no-check-certificate https://paddlehub.bj.bcebos.com/resources/test_img_cat.jpg
$ wget --no-check-certificate https://paddlehub.bj.bcebos.com/resources/test_img_dog.jpg
```

完整预测代码如下：
```python
import os
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

# Step 1: build Program
module_map = {
    "resnet50": "resnet_v2_50_imagenet",
    "resnet101": "resnet_v2_101_imagenet",
    "resnet152": "resnet_v2_152_imagenet",
    "mobilenet": "mobilenet_v2_imagenet",
    "nasnet": "nasnet_imagenet",
    "pnasnet": "pnasnet_imagenet"
}

module_name = module_map["resnet50"]
module = hub.Module(name = module_name)
input_dict, output_dict, program = module.context(trainable=False)
img = input_dict["image"]
feature_map = output_dict["feature_map"]

dataset = hub.dataset.DogCat()
task = hub.create_img_cls_task(
    feature=feature_map, num_classes=dataset.num_labels)
feed_list = [img.name]

# Step 2: create data reader
data = [
    "test_img_dog.jpg",
    "test_img_cat.jpg"
]

data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=None)

predict_reader = data_reader.data_generator(
    phase="predict", batch_size=1, data=data)

label_dict = dataset.label_dict()

# Step 3: switch to inference program
with fluid.program_guard(task.inference_program()):
    # Step 4: load pretrained parameters
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    pretrained_model_dir = os.path.join("cv_finetune_turtorial_demo", "best_model")
    fluid.io.load_persistables(exe, pretrained_model_dir)
    feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
    # Step 5: predict
    for index, batch in enumerate(predict_reader()):
        result, = exe.run(
            feed=feeder.feed(batch), fetch_list=[task.variable('probs')])
        predict_result = np.argsort(result[0])[::-1][0]
        print("input %i is %s, and the predict result is %s" %
              (index+1, data[index], label_dict[predict_result]))
```
