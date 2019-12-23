# PaddleHub 图像分类

本示例将展示如何使用PaddleHub Finetune API以及[ResNet](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_50_imagenet&en_category=ImageClassification)等预训练模型完成分类任务。

## 如何开始Finetune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_classifier.sh`即可开始使用ResNet对[Flowers](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Dataset#class-hubdatasetflowersdataset)等数据集进行Finetune。

其中脚本参数说明如下：

```shell
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数。默认为16
--num_epoch: finetune迭代的轮数。默认为1
--module: 使用哪个Module作为finetune的特征提取器，脚本支持{resnet50/resnet101/resnet152/mobilenet/nasnet/pnasnet}等模型。默认为resnet50
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型。默认为paddlehub_finetune_ckpt
--dataset: 使用什么数据集进行finetune, 脚本支持分别是{flowers/dogcat/stanforddogs/indoor67/food101}。默认为flowers
--use_gpu: 是否使用GPU进行训练，如果机器支持GPU且安装了GPU版本的PaddlePaddle，我们建议您打开这个开关。默认关闭
--use_data_parallel: 是否使用数据并行，打开该开关时，会将数据分散到不同的卡上进行训练（CPU下会分布到不同线程）。默认打开
```

## 代码步骤

使用PaddleHub Finetune API进行Finetune可以分为4个步骤

### Step1: 加载预训练模型

```python
module = hub.Module(name="resnet_v2_50_imagenet")
inputs, outputs, program = module.context(trainable=True)
```

PaddleHub提供许多图像分类预训练模型，如xception、mobilenet、efficientnet等，详细信息参见[图像分类模型](https://www.paddlepaddle.org.cn/hub?filter=en_category&value=ImageClassification)

### Step2: 下载数据集并使用ImageClassificationReader读取数据
```python
dataset = hub.dataset.Flowers()
data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)
```

其中数据集的准备代码可以参考 [flowers.py](../../paddlehub/dataset/flowers.py)
同时，PaddleHub提供了更多的图像分类数据集：

| 数据集    | API                                        |
| -------- | ------------------------------------------ |
| Flowers  | hub.dataset.Flowers()                      | 
| DogCat   | hub.dataset.DogCat()                       | 
| Indoor67 | hub.dataset.Indoor67()                     | 
| Food101  | hub.dataset.Food101()                      | 

`hub.dataset.Flowers()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录

`module.get_vocab_path()` 会返回预训练模型对应的词表

`max_seq_len` 需要与Step1中context接口传入的序列长度保持一致

MultiLabelClassifyReader中的`data_generator`会自动按照模型对应词表对数据进行tokenize，以迭代器的方式返回BERT所需要的Tensor格式，包括`input_ids`，`position_ids`，`segment_id`与序列对应的mask `input_mask`.

**NOTE**: Reader返回tensor的顺序是固定的，默认按照input_ids, position_ids, segment_id, input_mask这一顺序返回。

### Step3：选择优化策略和运行配置

```python
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_proportion=0.0,
    lr_scheduler="linear_decay",
)

config = hub.RunConfig(use_cuda=True, use_data_parallel=True, use_pyreader=True, num_epoch=3, batch_size=32, strategy=strategy)
```

#### 优化策略
针对ERNIE与BERT类任务，PaddleHub封装了适合这一任务的迁移学习优化策略`AdamWeightDecayStrategy`

* `learning_rate`: Finetune过程中的最大学习率;
* `weight_decay`: 模型的正则项参数，默认0.01，如果模型有过拟合倾向，可适当调高这一参数;
* `warmup_proportion`: 如果warmup_proportion>0, 例如0.1, 则学习率会在前10%的steps中线性增长至最高值learning_rate;
* `lr_scheduler`: 有两种策略可选(1) `linear_decay`策略学习率会在最高点后以线性方式衰减; `noam_decay`策略学习率会在最高点以多项式形式衰减；

#### 运行配置
`RunConfig` 主要控制Finetune的训练，包含以下可控制的参数:

* `log_interval`: 进度日志打印间隔，默认每10个step打印一次
* `eval_interval`: 模型评估的间隔，默认每100个step评估一次验证集
* `save_ckpt_interval`: 模型保存间隔，请根据任务大小配置，默认只保存验证集效果最好的模型和训练结束的模型
* `use_cuda`: 是否使用GPU训练，默认为False
* use_pyreader: 是否使用pyreader，默认False
* use_data_parallel: 是否使用并行计算，默认False。打开该功能依赖nccl库
* `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成
* `num_epoch`: finetune的轮数
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size
* `enable_memory_optim`: 是否使用内存优化， 默认为True
* `strategy`: Finetune优化策略

### Step4: 构建网络并创建分类迁移任务进行Finetune
```python
pooled_output = outputs["pooled_output"]

# feed_list的Tensor顺序不可以调整
feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.MultiLabelClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config)

cls_task.finetune_and_eval()
```
**NOTE:**
1. `outputs["pooled_output"]`返回了ERNIE/BERT模型对应的[CLS]向量,可以用于句子或句对的特征表达。
2. `feed_list`中的inputs参数指名了ERNIE/BERT中的输入tensor的顺序，与MultiLabelClassifierTask返回的结果一致。
3. `hub.MultiLabelClassifierTask`通过输入特征，label与迁移的类别数，可以生成适用于多标签分类的迁移任务`MultiLabelClassifierTask`

## 可视化

Finetune API训练过程中会自动对关键训练指标进行打点，启动程序后执行下面命令
```bash
$ tensorboard --logdir $CKPT_DIR/visualization --host ${HOST_IP} --port ${PORT_NUM}
```
其中${HOST_IP}为本机IP地址，${PORT_NUM}为可用端口号，如本机IP地址为192.168.0.1，端口号8040，用浏览器打开192.168.0.1:8040，即可看到训练过程中指标的变化情况

## 模型预测

通过Finetune完成模型训练后，在对应的ckpt目录下，会自动保存验证集上效果最好的模型。
配置脚本参数
```
CKPT_DIR="./ckpt_toxic"
python predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 128
```
其中CKPT_DIR为Finetune API保存最佳模型的路径, max_seq_len是ERNIE模型的最大序列长度，*请与训练时配置的参数保持一致*

参数配置正确后，请执行脚本`sh run_predict.sh`，即可看到以下文本分类预测结果, 以及最终准确率。
如需了解更多预测步骤，请参考`predict.py`


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
