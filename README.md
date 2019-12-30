# PaddleHub

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleHub.svg?branch=release/v1.3)](https://travis-ci.org/PaddlePaddle/PaddleHub)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)

PaddleHub是飞桨预训练模型管理和迁移学习工具，通过PaddleHub开发者可以使用高质量的预训练模型结合Fine-tune API快速完成迁移学习到应用部署的全流程工作。PaddleHub具有以下特性：

* 便捷获取飞桨生态下的高质量预训练模型，涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、视频分类、图像生成、图像分割、文本审核、关键点检测等主流模型。更多模型详情请查看官网：https://www.paddlepaddle.org.cn/hub
* 通过高质量预训练模型与PaddleHub Fine-tune API，只需要少量代码即可实现自然语言处理和计算机视觉场景的深度学习模型，更多Demo请参考以下链接：

  [文本分类](./demo/text_classification)     [序列标注](./demo/sequence_labeling)   [多标签分类](./demo/multi_label_classification)   [图像分类](./demo/image_classification) [检索式问答任务](./demo/qa_classification) [回归任务](./demo/regression)   [句子语义相似度计算](./demo/sentence_similarity) [阅读理解任务](./demo/reading_comprehension)

* 『**模型即软件**』的设计理念，通过Python API或命令行实现快速预测，更方便地使用PaddlePaddle模型库，更多介绍请参考教程[PaddleHub命令行工具介绍](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%B7%A5%E5%85%B7)
* PaddleHub提供便捷的服务化部署能力，简单一行命令即可搭建属于自己的模型的API服务，更多详情请参考教程[PaddleHub Serving一键服务化部署](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Serving%E4%B8%80%E9%94%AE%E6%9C%8D%E5%8A%A1%E9%83%A8%E7%BD%B2)和[使用示例](./demo/serving)
* 支持AutoDL Finetuner超参优化技术， 自动搜索最优模型超参得到更好的模型效果。详情请参考[AutoDL Finetuner超参优化功能教程](./tutorial/autofinetune.md)

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
2. 使用PaddleHub下载数据集、预训练模型等，要求机器可以访问外网。可以使用server_check()可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully.
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully.
```


## 快速体验
安装成功后，执行命令[hub run](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%B7%A5%E5%85%B7#run)，可以快速体验PaddleHub无需代码、一键预测的命令行功能，如下三个示例：

使用[词法分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=LexicalAnalysis)模型LAC进行分词
```shell
$ hub run lac --input_text "今天是个好日子"
[{'word': ['今天', '是', '个', '好日子'], 'tag': ['TIME', 'v', 'q', 'n']}]
```

使用[情感分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=SentimentAnalysis)模型Senta对句子进行情感预测
```shell
$ hub run senta_bilstm --input_text "今天天气真好"
{'text': '今天天气真好', 'sentiment_label': 1, 'sentiment_key': 'positive', 'positive_probs': 0.9798, 'negative_probs': 0.0202}]
```

使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型 SSD/YOLO v3/Faster RCNN 对图片进行目标检测
```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_object_detection.jpg
$ hub run ssd_mobilenet_v1_pascal --input_path test_object_detection.jpg
$ hub run yolov3_darknet53_coco2017 --input_path test_object_detection.jpg
$ hub run faster_rcnn_coco2017 --input_path test_object_detection.jpg
```
![SSD检测结果](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.3/docs/imgs/object_detection_result.png)

除了上述三类模型外，PaddleHub还发布了图像分类、语义模型、视频分类、图像生成、图像分割、文本审核、关键点检测等业界主流模型，更多PaddleHub已经发布的模型，请前往 https://www.paddlepaddle.org.cn/hub 查看

## 教程

我们在AI Studio上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|ResNet|图像分类|猫狗数据集DogCat|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216772)||
|ERNIE|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216764)||
|ERNIE|文本分类|中文新闻分类数据集THUNEWS|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216649)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成文本分类迁移学习。|
|ERNIE|序列标注|中文序列标注数据集MSRA_NER|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216787)||
|ERNIE|序列标注|中文快递单数据集Express|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216683)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成序列标注迁移学习。|
|ERNIE Tiny|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215599)||
|Senta|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216851)|本教程讲述了任何利用Senta和Fine-tune API完成情感分类迁移学习。|
|Senta|情感分析预测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216735)||
|LAC|词法分析|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215641)||
|Ultra-Light-Fast-Generic-Face-Detector-1MB|人脸检测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216749)||


同时，关于PaddleHub更多信息参考：

[Fine-tune API](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Finetune-API)

[自定义数据集如何Fine-tune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)

[实现自定义迁移任务](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub:-%E8%87%AA%E5%AE%9A%E4%B9%89Task)

[PaddleHub Serving一键服务化部署](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Serving%E4%B8%80%E9%94%AE%E6%9C%8D%E5%8A%A1%E9%83%A8%E7%BD%B2)

[自动优化超参AutoDL Finetuner使用教程](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.3/tutorial/autofinetune.md)

[迁移学习与ULMFiT微调策略](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.3/tutorial/strategy_exp.md)



## FAQ

**Q:** 利用PaddleHub Fine-tune如何适配自定义数据集

**A:** 参考[PaddleHub适配自定义数据集完成Fine-tune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)


**Q:** 使用PaddleHub时，无法下载预置数据集、Module的等现象

**A:** 下载数据集、module等，PaddleHub要求机器可以访问外网。可以使用server_check()可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully.
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully.
```

**Q:** 利用PaddleHub ernie/bert进行Fine-tune时，运行出错并提示`paddle.fluid.core_avx.EnforceNotMet: Input ShapeTensor cannot be found in Op reshape2`等信息

**A:** 因为ernie/bert module的创建时和此时运行环境中PaddlePaddle版本不对应。可以将PaddlePaddle和PaddleHub升级至最新版本，同时将ernie卸载。
```shell
$ pip install --upgrade paddlehub
$ hub uninstall ernie
```


**NOTE**： PaddleHub 1.1.1版本已支持离线运行Module


**更多问题**

当安装或者使用遇到问题时，可以通过[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)查找解决方案。
如果在FAQ中没有找到解决方案，欢迎您将问题和bug报告以[Github Issues](https://github.com/PaddlePaddle/PaddleHub/issues)的形式提交给我们，我们会第一时间进行跟进

## 用户交流群

* 飞桨PaddlePaddle 交流群：796771754（QQ群）
* 飞桨 ERNIE交流群：760439550（QQ群）


## 更新历史

PaddleHub v1.4.1已发布！

详情参考[更新历史](./RELEASE.md)
