# PaddleHub 图像分类

本示例将展示如何使用PaddleHub Fine-tune API以及[ResNet](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_50_imagenet&en_category=ImageClassification)等预训练模型完成分类任务。

## 如何开始Fine-tune

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_classifier.sh`即可开始使用ResNet对[Flowers](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Dataset#class-hubdatasetflowersdataset)等数据集进行Fine-tune。

其中脚本参数说明如下：

```shell
--batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数。默认为16；
--num_epoch: Fine-tune迭代的轮数。默认为1；
--module: 使用哪个Module作为Fine-tune的特征提取器，脚本支持{resnet50/resnet101/resnet152/mobilenet/nasnet/pnasnet}等模型。默认为resnet50；
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型。默认为paddlehub_finetune_ckpt；
--dataset: 使用什么数据集进行Fine-tune, 脚本支持分别是{flowers/dogcat/stanforddogs/indoor67/food101}。默认为flowers；
--use_gpu: 是否使用GPU进行训练，如果机器支持GPU且安装了GPU版本的PaddlePaddle，我们建议您打开这个开关。默认关闭；
--use_data_parallel: 是否使用数据并行，打开该开关时，会将数据分散到不同的卡上进行训练（CPU下会分布到不同线程）。默认打开；
```

## 代码步骤

使用PaddleHub Fine-tune API进行Fine-tune可以分为4个步骤。

### Step1: 加载预训练模型

```python
module = hub.Module(name="resnet_v2_50_imagenet")
inputs, outputs, program = module.context(trainable=True)
```

PaddleHub提供许多图像分类预训练模型，如xception、mobilenet、efficientnet等，详细信息参见[图像分类模型](https://www.paddlepaddle.org.cn/hub?filter=en_category&value=ImageClassification)。

如果想尝试efficientnet模型，只需要更换Module中的`name`参数即可.
```python
# 更换name参数即可无缝切换efficientnet模型, 代码示例如下
module = hub.Module(name="efficientnetb7_imagenet")
```

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

`hub.dataset.Flowers()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。

`module.get_expected_image_width()` 和 `module.get_expected_image_height()`会返回预训练模型对应的图片尺寸。

`module.module.get_pretrained_images_mean()` 和 `module.get_pretrained_images_std()`会返回预训练模型对应的图片均值和方差。

#### 自定义数据集

如果想加载自定义数据集完成迁移学习，详细参见[自定义数据集](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)。

### Step3：选择优化策略和运行配置

```python
strategy = hub.DefaultFinetuneStrategy(
    learning_rate=1e-4,
    optimizer_name="adam",
    regularization_coeff=1e-3)

config = hub.RunConfig(use_cuda=True, use_data_parallel=True, num_epoch=3, batch_size=32, strategy=strategy)
```

#### 优化策略

PaddleHub提供了许多优化策略，如`AdamWeightDecayStrategy`、`ULMFiTStrategy`、`DefaultFinetuneStrategy`等，详细信息参见[策略](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Strategy)。

其中`DefaultFinetuneStrategy`:

* `learning_rate`: 全局学习率。默认为1e-4；
* `optimizer_name`: 优化器名称。默认adam；
* `regularization_coeff`: 正则化的λ参数。默认为1e-3；

#### 运行配置
`RunConfig` 主要控制Fine-tune的训练，包含以下可控制的参数:

* `log_interval`: 进度日志打印间隔，默认每10个step打印一次；
* `eval_interval`: 模型评估的间隔，默认每100个step评估一次验证集；
* `save_ckpt_interval`: 模型保存间隔，请根据任务大小配置，默认只保存验证集效果最好的模型和训练结束的模型；
* `use_cuda`: 是否使用GPU训练，默认为False；
* `use_pyreader`: 是否使用pyreader，默认False；
* `use_data_parallel`: 是否使用并行计算，默认True。打开该功能依赖nccl库；
* `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
* `num_epoch`: Fine-tune的轮数；
* `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* `strategy`: Fine-tune优化策略；

### Step4: 构建网络并创建分类迁移任务进行Fine-tune

```python
feature_map = output_dict["feature_map"]
feed_list = [input_dict["image"].name]

task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)

task.finetune_and_eval()
```
**NOTE:**
1. `output_dict["feature_map"]`返回了resnet/mobilenet等模型对应的feature_map，可以用于图片的特征表达。
2. `feed_list`中的inputs参数指明了resnet/mobilenet等模型的输入tensor的顺序，与ImageClassifierTask返回的结果一致。
3. `hub.ImageClassifierTask`通过输入特征，label与迁移的类别数，可以生成适用于图像分类的迁移任务`ImageClassifierTask`。

#### 自定义迁移任务

如果想改变迁移任务组网，详细参见[自定义迁移任务](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub:-%E8%87%AA%E5%AE%9A%E4%B9%89Task)。

## 可视化

Fine-tune API训练过程中会自动对关键训练指标进行打点，启动程序后执行下面命令
```bash
$ visualdl --logdir $CKPT_DIR/visualization --host ${HOST_IP} --port ${PORT_NUM}
```
其中${HOST_IP}为本机IP地址，${PORT_NUM}为可用端口号，如本机IP地址为192.168.0.1，端口号8040，用浏览器打开192.168.0.1:8040，即可看到训练过程中指标的变化情况。

## 模型预测

当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为Fine-tune时所选择的保存checkpoint的目录。

我们使用该模型来进行预测。predict.py脚本支持的参数如下：

```shell
--module: 使用哪个Module作为Fine-tune的特征提取器，脚本支持{resnet50/resnet101/resnet152/mobilenet/nasnet/pnasnet}等模型。默认为resnet50；
--checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型。默认为paddlehub_finetune_ckpt；
--dataset: 使用什么数据集进行Fine-tune, 脚本支持分别是{flowers/dogcat}。默认为flowers；
--use_gpu: 使用使用GPU进行训练，如果本机支持GPU且安装了GPU版本的PaddlePaddle，我们建议您打开这个开关。默认关闭；
--use_pyreader: 是否使用pyreader进行数据喂入。默认关闭；
```

**NOTE:** 进行预测时，所选择的module，checkpoint_dir，dataset必须和Fine-tune所用的一样。

参数配置正确后，请执行脚本`sh run_predict.sh`，即可看到以下图片分类预测结果。
如需了解更多预测步骤，请参考`predict.py`。

我们在AI Studio上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|ResNet|图像分类|猫狗数据集DogCat|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147010)||
|ERNIE|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147006)||
|ERNIE|文本分类|中文新闻分类数据集THUNEWS|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/221999)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成文本分类迁移学习。|
|ERNIE|序列标注|中文序列标注数据集MSRA_NER|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147009)||
|ERNIE|序列标注|中文快递单数据集Express|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/184200)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成序列标注迁移学习。|
|ERNIE Tiny|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/186443)||
|Senta|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216846)|本教程讲述了任何利用Senta和Fine-tune API完成情感分类迁移学习。|
|Senta|情感分析预测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215814)||
|LAC|词法分析|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215711)||
|Ultra-Light-Fast-Generic-Face-Detector-1MB|人脸检测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215962)||


## 超参优化AutoDL Finetuner

PaddleHub还提供了超参优化（Hyperparameter Tuning）功能， 自动搜索最优模型超参得到更好的模型效果。详细信息参见[AutoDL Finetuner超参优化功能教程](../../docs/tutorial/autofinetune.md)。
