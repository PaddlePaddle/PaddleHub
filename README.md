<p align="center">
 <img src="./docs/imgs/paddlehub_logo.jpg" align="middle"  
</p>

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleHub.svg?branch=release/v1.3)](https://travis-ci.org/PaddlePaddle/PaddleHub)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)

PaddleHub是飞桨生态的预训练模型应用工具，开发者可以便捷地使用高质量的预训练模型结合Fine-tune API快速完成模型迁移到部署的全流程工作。PaddleHub提供的预训练模型涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、视频分类、图像生成、图像分割、文本审核、关键点检测等主流模型。更多详情可查看官网：https://www.paddlepaddle.org.cn/hub **目前最新版本为1.6.0**。


PaddleHub以预训练模型为核心具备以下特点：  

* **[模型即软件](#模型即软件)**，通过Python API或命令行实现模型调用，可快速体验或集成飞桨特色预训练模型。

* **[易用的迁移学习](#迁移学习)**，通过Fine-tune API，内置多种优化策略，只需少量代码即可完成预训练模型的Fine-tuning。

* **[一键模型转服务](#服务化部署paddlehub-serving)**，简单一行命令即可搭建属于自己的深度学习模型API服务。

* **[自动超参优化](#超参优化autodl-finetuner)**，内置AutoDL Finetuner能力，一键启动自动化超参搜索。


<p align="center">
 <img src="./docs/imgs/paddlehub_finetune.gif" align="middle"  
</p>

<p align='center'>
 十行代码完成ERNIE工业级文本分类
</p>


## 目录

* [安装](#%E5%AE%89%E8%A3%85)
* [特性](#特性)
* [FAQ](#faq)
* [用户交流群](#%E7%94%A8%E6%88%B7%E4%BA%A4%E6%B5%81%E7%BE%A4)
* [更新历史](#%E6%9B%B4%E6%96%B0%E5%8E%86%E5%8F%B2)


## 安装

### 环境依赖

* Python>=3.6 
* PaddlePaddle>=1.6.1

除上述依赖外，预训练模型和数据集的下载需要网络连接，请确保机器可以正常访问网络。若本地已存在相关的数据集和预训练模型，则可以离线运行PaddleHub。

## 特性

### 模型即软件

PaddleHub采用 **模型即软件** 的理念，通过Python API或命令行实现快速预测，更方便地使用PaddlePaddle预训练模型。
安装PaddleHub成功后，执行命令[hub run](./docs/tutorial/cmdintro.md)，可以快速体验PaddleHub无需代码、一键预测的命令行功能，如下三个示例：

* 使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型pyramidbox_lite_mobile_mask对图片进行口罩检测
```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_mask_detection.jpg
$ hub run pyramidbox_lite_mobile_mask --input_path test_mask_detection.jpg
```
![人脸识别结果](docs/imgs/test_mask_detection_result.jpg)

* 使用[词法分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=LexicalAnalysis)模型LAC进行分词
```shell
$ hub run lac --input_text "今天是个好日子"
[{'word': ['今天', '是', '个', '好日子'], 'tag': ['TIME', 'v', 'q', 'n']}]
```

* 使用[情感分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=SentimentAnalysis)模型Senta对句子进行情感预测
```shell
$ hub run senta_bilstm --input_text "今天天气真好"
{'text': '今天天气真好', 'sentiment_label': 1, 'sentiment_key': 'positive', 'positive_probs': 0.9798, 'negative_probs': 0.0202}]
```

* 使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型Ultra-Light-Fast-Generic-Face-Detector-1MB对图片进行人脸识别
```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
$ hub run ultra_light_fast_generic_face_detector_1mb_640 --input_path test_image.jpg
```
![人脸识别结果](docs/imgs/face_detection_result.jpeg)

* 使用[图像分割](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=ImageSegmentation)模型对进行人像扣图和人体部件识别
``、`shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
$ hub run ace2p --input_path test_image.jpg
$ hub run deeplabv3p_xception65_humanseg --input_path test_image.jpg
```
![人体部件分割结果](docs/imgs/img_seg_result.jpeg)

PaddleHub还提供图像分类、语义模型、视频分类、图像生成、图像分割、文本审核、关键点检测等主流模型，更多模型介绍，请前往 [https://www.paddlepaddle.org.cn/hub](https://www.paddlepaddle.org.cn/hub) 查看

### 易用的迁移学习

通过PaddleHub的Fine-tune API，只需要少量代码即可完成深度学习模型在自然语言处理和计算机视觉场景下的迁移学习。

* 示例合集

  PaddleHub提供了使用Fine-tune API和预训练模型完成[文本分类](./demo/text_classification)、[序列标注](./demo/sequence_labeling)、[多标签分类](./demo/multi_label_classification)、[图像分类](./demo/image_classification)、[检索式问答任务](./demo/qa_classification)、[回归任务](./demo/regression)、[句子语义相似度计算](./demo/sentence_similarity)、[阅读理解任务](./demo/reading_comprehension)等迁移任务的使用示例，详细参见[demo](./demo)。

* AI Studio教程

  PaddleHub在AI Studio上提供了IPython Notebook形式的demo。用户可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|pyramidbox_lite_mobile_mask|口罩检测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/267322)|
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

**NOTE:** [`飞桨PaddleHub`](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/79927)是AI Studio的官方账号。

更多Fine-tune API的使用教程可参考：

[Fine-tune API](./docs/reference)

[如何对自定义数据集进行Fine-tuning](./docs/tutorial/how_to_load_data.md)

[如何自定义迁移任务](./docs/tutorial/how_to_define_task.md)

[ULMFiT优化策略](./docs/tutorial/strategy_exp.md)

### 一键模型转服务

PaddleHub提供便捷的模型转服务的能力，只需简单一行命令即可实现预训练模型的HTTP服务部署。

**PaddleHub 1.5.0版本增加文本Embedding服务[Bert Service](./docs/tutorial/bert_service.md), 轻松获取文本embedding**

一键服务化启动方式有两种：

* 命令行方式：

```shell
$ hub serving start --modules [Module1==Version1, Module2==Version2, ...]
```

其中选项参数`--modules/-m`表示待部署模型。

* 配置文件方式：

```shell
$ hub serving start --config config.json
```

config.json文件包含待部署模型信息等，

关于模型服务化使用说明参见[PaddleHub模型服务化服务化部署](./docs/tutorial/serving.md)。

### 自动超参优化

PaddleHub内置AutoDL Finetuner能力，提供多种优化策略策略实现自动化超参搜索，使得模型在验证集上得到更好的结果，用户只需要一行命令`hub autofinetune`即可启动。更多详细使用说明请参见[PaddleHub超参优化](./docs/tutorial/autofinetune.md)。

## FAQ

**Q:** 利用PaddleHub Fine-tune如何适配自定义数据集？

**A:** 参考[PaddleHub适配自定义数据集完成Fine-tune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)。


**Q:** 使用PaddleHub时，无法下载预置数据集、Module的等现象。

**A:** 下载数据集、module等，PaddleHub要求机器可以访问外网。可以使用server_check()可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully。
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully。
```

**Q:** 利用PaddleHub ERNIE/BERT进行Fine-tune时，运行出错并提示`paddle.fluid.core_avx.EnforceNotMet: Input ShapeTensor cannot be found in Op reshape2`等信息。

**A:** 预训练模型版本与PaddlePaddle版本不匹配。可尝试将PaddlePaddle和PaddleHub升级至最新版本，并将原ERNIE模型卸载。
```shell
$ pip install --upgrade paddlehub
$ hub uninstall ernie
```

**FAQ**

当安装或者使用遇到问题时，可以通过[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)查找解决方案。
如果在FAQ中没有找到解决方案，欢迎您将问题以[Github Issues](https://github.com/PaddlePaddle/PaddleHub/issues)的形式提交给我们，我们会第一时间进行跟进。

## 用户交流群

* 飞桨PaddlePaddle 交流群：796771754（QQ群）
* 飞桨 ERNIE交流群：760439550（QQ群）


## 更新历史

PaddleHub v1.6.0已发布！

详情参考[更新历史](./RELEASE.md)
