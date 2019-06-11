# PaddleHub 图像分类

## 关于

本示例将展示如何使用PaddleHub Finetune API以及[图像分类](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)预训练模型完成分类任务。

## 准备工作

在运行本目录的脚本前，需要先安装1.3.0版本以上的PaddlePaddle（如果您本地已经安装了符合条件的PaddlePaddle版本，那么可以跳过`准备工作`这一步）。

如果您的机器支持GPU，我们建议下载GPU版本的PaddlePaddle，使用GPU进行训练和预测的效率都比使用CPU要高。
```shell
# 安装GPU版本的PaddlePaddle
$ pip install --upgrade paddlepaddle-gpu
```

如果您的机器不支持GPU，可以通过下面的命令来安装CPU版本的PaddlePaddle

```shell
# 安装CPU版本的PaddlePaddle
$ pip install --upgrade paddlepaddle
```

在安装过程中如果遇到问题，您可以到[Paddle官方网站](http://www.paddlepaddle.org/)上查看解决方案。

## 开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_classifier.sh`即可开始使用进行finetune。

脚本支持的参数如下：

```shell
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数。默认为16
--num_epoch: finetune迭代的轮数。默认为1
--module: 使用哪个Module作为finetune的特征提取器，脚本支持{resnet50/resnet101/resnet152/mobilenet/nasnet/pnasnet}等模型。默认为resnet50
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型。默认为paddlehub_finetune_ckpt
--dataset: 使用什么数据集进行finetune, 脚本支持分别是{flowers/dogcat/stanforddogs/indoor67/food101}。默认为flowers
--use_gpu: 是否使用GPU进行训练，如果机器支持GPU且安装了GPU版本的PaddlePaddle，我们建议您打开这个开关。默认关闭
--use_data_parallel: 是否使用数据并行，打开该开关时，会将数据分散到不同的卡上进行训练（CPU下会分布到不同线程）。默认关闭
--use_pyreader: 是否使用pyreader进行数据喂入。默认关闭
```

## 进行预测

当完成finetune后，finetune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为finetune时所选择的保存checkpoint的目录。

我们使用该模型来进行预测。执行脚本`sh predict.sh`即可开始使用进行预测。

脚本支持的参数如下：

```shell
--module: 使用哪个Module作为finetune的特征提取器，脚本支持{resnet50/resnet101/resnet152/mobilenet/nasnet/pnasnet}等模型。默认为resnet50
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型。默认为paddlehub_finetune_ckpt
--dataset: 使用什么数据集进行finetune, 脚本支持分别是{flowers/dogcat}。默认为flowers
--use_gpu: 使用使用GPU进行训练，如果本机支持GPU且安装了GPU版本的PaddlePaddle，我们建议您打开这个开关。默认关闭
--use_pyreader: 是否使用pyreader进行数据喂入。默认关闭
```

`注意`：进行预测时，所选择的module，checkpoint_dir，dataset必须和finetune所用的一样
