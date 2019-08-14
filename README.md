# PaddleHub

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleHub.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/PaddleHub)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)

PaddleHub是基于PaddlePaddle生态下的预训练模型管理和迁移学习工具，可以结合预训练模型更便捷地开展迁移学习工作。通过PaddleHub，您可以：

* 便捷地获取PaddlePaddle生态下的所有预训练模型，涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、语言模型、视频分类、图像生成八类主流模型。
  * 更多详情可查看官网：http://hub.paddlepaddle.org.cn
* 通过PaddleHub Fine-tune API，结合少量代码即可完成**大规模预训练模型**的迁移学习，具体Demo可参考以下链接：
  * [文本分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.0.0/demo/text-classification)
  * [序列标注](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.0.0/demo/sequence-labeling)
  * [多标签分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.0.0/demo/multi-label-classification)
  * [图像分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.0.0/demo/image-classification)
  * [检索式问答任务](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.0.0/demo/qa_classification)
* PaddleHub引入『**模型即软件**』的设计理念，支持通过Python API或者命令行工具，一键完成预训练模型地预测，更方便的应用PaddlePaddle模型库。
  * [PaddleHub命令行工具介绍](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%B7%A5%E5%85%B7)

## 环境依赖
* Python==2.7 or Python>=3.5
* PaddlePaddle>=1.4.0

除上述依赖外，PaddleHub的预训练模型和预置数据集需要连接服务端进行下载，请确保机器可以正常访问网络

## 常见问题

### 问题一

**现象**

利用PaddleHub ernie/bert进行Finetune时，提示`paddle.fluid.core_avx.EnforceNotMet: Input ShapeTensor cannot be found in Op reshape2`等信息

**原因**

这是因为ernie/bert module的创建时和此时运行环境中PaddlePaddle版本不对应。

**解决方法**

首先将PaddlePaddle和PaddleHub升级至最新版本，同时将ernie卸载。

如果机器不支持GPU，那么使用如下命令来安装PaddlePaddle的CPU版本
```shell
$ pip install --upgrade paddlepaddle
$ pip install --upgrade paddlehub
$ hub uninstall ernie
```

如果机器支持GPU，则使用如下命令来安装PaddlePaddle的GPU版本
```shell
$ pip install --upgrade paddlepaddle-gpu
$ pip install --upgrade paddlehub
$ hub uninstall ernie
```

### 问题二

**现象**

使用paddlehub时，无法下载预置数据集、Module的等现象

**原因**

PaddleHub中的预训练模型和预置数据集都需要通过服务端进行下载，因此Paddle Hub默认用户访问外网权限。
可以通过以下命令确认是否可以访问外网。

```python
import requests

res = requests.get('http://paddlepaddle.org.cn/paddlehub/search', {'word': 'ernie', 'type': 'Module'})
print(res)

# the common result is like this:
# <Response [200]>
```

**Note**

PaddleHub 1.1.1版本已支持离线运行Module

### 问题三

**现象**

利用PaddleHub Finetune如何适配自定义数据集

**解决方法**

参考[PaddleHub Finetune适配自定义数据集完成Finetune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)

## 答疑

当安装或者使用遇到问题时，可以通过[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)查找解决方案。
如果在FAQ中没有找到解决方案，欢迎您将问题和bug报告以[Github Issues](https://github.com/PaddlePaddle/PaddleHub/issues)的形式提交给我们，我们会第一时间进行跟进

## 安装
pip安装方式如下：

```shell
$ pip install paddlehub
```

## 快速体验
安装成功后，执行下面的命令，可以快速体验PaddleHub无需代码、一键预测的命令行功能：

`示例一`

使用[词法分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=LexicalAnalysis)模型LAC进行分词
```shell
$ hub run lac --input_text "今天是个好日子"
[{'word': ['今天', '是', '个', '好日子'], 'tag': ['TIME', 'v', 'q', 'n']}]
```

`示例二`

使用[情感分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=SentimentAnalysis)模型Senta对句子进行情感预测
```shell
$ hub run senta_bilstm --input_text "今天是个好日子"
[{'text': '今天是个好日子', 'sentiment_label': 2, 'sentiment_key': 'positive', 'positive_probs': 0.6065, 'negative_probs': 0.3935}]
```

`示例三`

使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型 SSD/YOLO v3/Faster RCNN 对图片进行目标检测
```shell
$ wget --no-check-certificate https://paddlehub.bj.bcebos.com/resources/test_object_detection.jpg
$ hub run ssd_mobilenet_v1_pascal --input_path test_object_detection.jpg
$ hub run yolov3_coco2017 --input_path test_object_detection.jpg
$ hub run faster_rcnn_coco2017 --input_path test_object_detection.jpg
```
![SSD检测结果](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.0.0/docs/imgs/object_detection_result.png)

除了上述三类模型外，PaddleHub还发布了语言模型、语义模型、图像分类、生成模型、视频分类等业界主流模型，更多PaddleHub已经发布的模型，请前往 http://hub.paddlepaddle.org.cn 查看

## 教程

[API](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Finetune-API)

[迁移学习](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E4%B8%8E%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)

[自定义迁移任务](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub:-%E8%87%AA%E5%AE%9A%E4%B9%89Task)

## 在线体验
我们在AI Studio和AIBook上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：
* ERNIE文本分类:
  * [AI Studio](https://aistudio.baidu.com/aistudio/projectDetail/79380)
  * [AIBook](https://console.bce.baidu.com/bml/?_=1562072915183#/bml/aibook/ernie_txt_cls)
* ERNIE序列标注:
  * [AI Studio](https://aistudio.baidu.com/aistudio/projectDetail/79377)
  * [AIBook](https://console.bce.baidu.com/bml/?_=1562072915183#/bml/aibook/ernie_seq_label)
* ELMo文本分类:
  * [AI Studio](https://aistudio.baidu.com/aistudio/projectDetail/79400)
  * [AIBook](https://console.bce.baidu.com/bml/#/bml/aibook/elmo_txt_cls)
* senta情感分类:
  * [AI Studio](https://aistudio.baidu.com/aistudio/projectDetail/79398)
  * [AIBook](https://console.bce.baidu.com/bml/#/bml/aibook/senta_bilstm)
* 图像分类:
  * [AI Studio](https://aistudio.baidu.com/aistudio/projectDetail/79378)
  * [AIBook](https://console.bce.baidu.com/bml/#/bml/aibook/img_cls)

## 更新历史

### PaddleHub v1.1.1

* PaddleHub支持离线运行Module
* 修复python2安装PaddleHub失败问题

### PaddleHub v1.1.0

* PaddleHub 新增 ERNIE 2.0
  * 升级Reader， 支持自动传送数据给Ernie 1.0/2.0
  * 新增数据集GLUE(MRPC、QQP、SST-2、CoLA、QNLI、RTE、MNLI)

### PaddleHub v1.0.1

* 安装模型时自动选择与paddlepaddle版本适配的模型

### PaddleHub v1.0.0

* 全新发布PaddleHub官网，易用性全面提升
  * 新增网站  http://hub.paddlepaddle.org.cn  包含PaddlePaddle生态的预训练模型使用介绍
  * 迁移学习Demo接入AI Studio与AI Book,无需安装即可快速体验

* 新增29个预训练模型，覆盖文本、图像、视频三大领域；目前官方提供40个预训练模型
  * CV预训练模型：
    * 新增图像分类预训练模型11个：SE_ResNeXt, GoogleNet, ShuffleNet等
    * 新增目标检测模型Faster-RCNN和YOLOv3
    * 新增图像生成模型CycleGAN
    * 新增人脸检测模型Pyramidbox
    * 新增视频分类模型4个: TSN, TSM, StNet, Non-Local
  * NLP预训练模型
    * 新增语义模型ELMo
    * 新增情感分析模型5个: Senta-BOW, Senta-CNN, Senta-GRNN, , Senta-LSTM, EmoTect
    * 新增中文语义相似度分析模型SimNet
    * 升级LAC词法分析模型，新增词典干预功能，支持用户自定义分词
* Fine-tune API升级，灵活性与性能全面提升
  * 支持多卡并行、PyReader多线程IO，Fine-tune速度提升60%
  * 简化finetune、evaluate、predict等使用逻辑，提升易用性
  * 增加事件回调功能，方便用户快速实现自定义迁移学习任务
  * 新增多标签分类Fine-tune任务

### PaddleHub v0.5.0

正式发布PaddleHub预训练模型管理工具，旨在帮助用户更高效的管理模型并开展迁移学习的工作。

* 预训练模型管理: 通过hub命令行可完成PaddlePaddle生态的预训练模型下载、搜索、版本管理等功能。

* 命令行一键使用: 无需代码，通过命令行即可直接使用预训练模型进行预测，快速调研训练模型效果。
                目前版本支持以下模型：词法分析LAC；情感分析Senta；目标检测SSD；图像分类ResNet, MobileNet, NASNet等。

* 迁移学习: 提供了基于预训练模型的Finetune API，用户通过少量代码即可完成迁移学习，包括BERT/ERNIE文本分类、序列标注、图像分类迁移等。


## 版权和许可证
PaddleHub由[Apache-2.0 license](LICENSE)提供
