# PaddleHub

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleHub.svg?branch=release/v1.3)](https://travis-ci.org/PaddlePaddle/PaddleHub)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)

PaddleHub是基于PaddlePaddle生态下的预训练模型管理和迁移学习工具，可以结合预训练模型更便捷地开展迁移学习工作。PaddleHub特性：

* 便捷地获取PaddlePaddle生态下的所有预训练模型，涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、语言模型、视频分类、图像生成、图像分割等主流模型。
  * 更多详情可查看官网：https://www.paddlepaddle.org.cn/hub
* 通过PaddleHub Fine-tune API，结合少量代码即可完成**大规模预训练模型**的迁移学习，具体Demo可参考以下链接：
  * [文本分类](./demo/text-classification)
  * [序列标注](./demo/sequence-labeling)
  * [多标签分类](./demo/multi-label-classification)
  * [图像分类](./demo/image-classification)
  * [检索式问答任务](./demo/qa_classification)
  * [回归任务](./demo/sentence_similarity)
  * [句子语义相似度计算](./demo/sentence_similarity)
  * [阅读理解任务](./demo/reading-comprehension)
* 支持超参优化（AutoDL Finetuner），自动调整超参数，给出效果较佳的超参数组合。
  * [PaddleHub超参优化功能AutoDL Finetuner使用示例](./demo/autofinetune)
* 引入『**模型即软件**』的设计理念，通过Python API或者命令行实现一键预测，更方便地应用PaddlePaddle模型库。
  * [PaddleHub命令行工具介绍](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%B7%A5%E5%85%B7)
* 一键Module服务化部署 - HubServing
  * [PaddleHub-Serving一键服务部署](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Serving%E4%B8%80%E9%94%AE%E6%9C%8D%E5%8A%A1%E9%83%A8%E7%BD%B2)
  * [使用示例](./demo/serving)

## 目录

* [安装](#%E5%AE%89%E8%A3%85)
* [快速体验](#%E5%BF%AB%E9%80%9F%E4%BD%93%E9%AA%8C)
* [教程](#%E6%95%99%E7%A8%8B)
* [FAQ](#faq)
* [用户交流群](#%E7%94%A8%E6%88%B7%E4%BA%A4%E6%B5%81%E7%BE%A4)
* [更新历史](#%E6%9B%B4%E6%96%B0%E5%8E%86%E5%8F%B2)


## 安装

### 环境依赖
* Python==2.7 or Python>=3.5 for Linux or Mac

  **Python>=3.6 for Windows**

* PaddlePaddle>=1.5

除上述依赖外，PaddleHub的预训练模型和预置数据集需要连接服务端进行下载，请确保机器可以正常访问网络。若本地已存在相关的数据集和预训练模型，则可以离线运行PaddleHub。

**NOTE:**
1. 若是出现离线运行PaddleHub错误，请更新PaddleHub 1.1.1版本之上。
pip安装方式如下：

```shell
$ pip install paddlehub
```
2. 下载数据集、module等，PaddleHub要求机器可以访问外网。可以使用server_check()可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully.
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully.
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
$ hub run senta_bilstm --input_text "今天天气真好"
{'text': '今天天气真好', 'sentiment_label': 1, 'sentiment_key': 'positive', 'positive_probs': 0.9798, 'negative_probs': 0.0202}]
```

`示例三`

使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型 SSD/YOLO v3/Faster RCNN 对图片进行目标检测
```shell
$ wget --no-check-certificate https://paddlehub.bj.bcebos.com/resources/test_object_detection.jpg
$ hub run ssd_mobilenet_v1_pascal --input_path test_object_detection.jpg
$ hub run yolov3_coco2017 --input_path test_object_detection.jpg
$ hub run faster_rcnn_coco2017 --input_path test_object_detection.jpg
```
![SSD检测结果](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.3/docs/imgs/object_detection_result.png)

除了上述三类模型外，PaddleHub还发布了语言模型、语义模型、图像分类、生成模型、视频分类等业界主流模型，更多PaddleHub已经发布的模型，请前往 https://www.paddlepaddle.org.cn/hub 查看

同时，我们在AI Studio上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：

|类别|AIStudio链接|
|-|-|
|ERNIE Tiny 文本分类|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/186443)
|ERNIE文本分类|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/79380)|
|ERNIE序列标注|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/79377)|
|ELMo文本分类|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/79400)|
|senta情感分类|[点击体验](https://aistudio.baidu.com/aistudio/projectDetail/79398)|
|图像分类| [点击体验](https://aistudio.baidu.com/aistudio/projectDetail/79378)|
|自定义数据FineTune（序列标注任务）|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/184200)|
|自定义数据FineTune（文本分类任务）|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/185121) |


## 教程

PaddleHub Fine-tune API 详情参考[wiki教程](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Finetune-API)

PaddleHub如何完成迁移学习，详情参考[wiki教程](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E4%B8%8E%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)

PaddleHub如何自定义迁移任务，详情参考[wiki教程](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub:-%E8%87%AA%E5%AE%9A%E4%B9%89Task)

PaddleHub如何自动优化超参数，详情参考[AutoDL Finetuner使用教程](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.3/tutorial/autofinetune.md)

PaddleHub如何使用ULMFiT策略微调预训练模型，详情参考[PaddleHub 迁移学习与ULMFiT微调策略](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.3/tutorial/strategy_exp.md)

## FAQ

**Q:** 利用PaddleHub ernie/bert进行Finetune时，运行出错并提示`paddle.fluid.core_avx.EnforceNotMet: Input ShapeTensor cannot be found in Op reshape2`等信息

**A:** 因为ernie/bert module的创建时和此时运行环境中PaddlePaddle版本不对应。可以将PaddlePaddle和PaddleHub升级至最新版本，同时将ernie卸载。
```shell
$ pip install --upgrade paddlehub
$ hub uninstall ernie
```

**Q:** 使用PaddleHub时，无法下载预置数据集、Module的等现象

**A:** PaddleHub中的预训练模型和预置数据集都需要通过服务端进行下载，因此PaddleHub默认用户访问外网权限。
可以通过以下命令确认是否可以访问外网。

```python
import requests

res = requests.get('http://paddlepaddle.org.cn/paddlehub/search', {'word': 'ernie', 'type': 'Module'})
print(res)

# the common result is like this:
# <Response [200]>
```
**Note：** PaddleHub 1.1.1版本已支持离线运行Module

**Q:** 利用PaddleHub Finetune如何适配自定义数据集

**A:** 参考[PaddleHub适配自定义数据集完成Finetune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)


**更多问题**

当安装或者使用遇到问题时，可以通过[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)查找解决方案。
如果在FAQ中没有找到解决方案，欢迎您将问题和bug报告以[Github Issues](https://github.com/PaddlePaddle/PaddleHub/issues)的形式提交给我们，我们会第一时间进行跟进

## 用户交流群

* 飞桨PaddlePaddle 交流群：796771754（QQ群）
* 飞桨 ERNIE交流群：760439550（QQ群）


## 更新历史

详情参考[更新历史](./RELEASE.md)
