# PaddleHub

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleHub.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/PaddleHub)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)

PaddleHub是基于PaddlePaddle开发的预训练模型管理工具，可以借助预训练模型更便捷地开展迁移学习工作。

## 特性

通过PaddleHub，您可以：

1. 通过命令行，无需编写代码，一键使用预训练模型进行预测；
2. 通过hub download命令，快速地获取PaddlePaddle生态下的所有预训练模型；
3. 借助PaddleHub Finetune API，使用少量代码完成迁移学习；
   - 更多Demo可参考 [ERNIE文本分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/demo/text-classification) [图像分类迁移](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/demo/image-classification)
   - 完整教程可参考 [文本分类迁移教程](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/nlp_tl_turtorial.md)  [图像分类迁移教程](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/cv_tl_turtorial.md)

## 安装

**环境依赖**
* Python==2.7 or Python>=3.5
* PaddlePaddle>=1.4.0

pip安装方式如下：

```shell
$ pip install paddlehub
```

## 快速体验

安装成功后，执行下面的命令，可以快速体验PaddleHub无需代码、一键预测的命令行功能：

```shell
# 使用百度LAC词法分析工具进行分词
$ hub run lac --input_text "今天是个好日子"

# 使用百度Senta情感分析模型对句子进行预测
$ hub run senta_bilstm --input_text "今天是个好日子"

# 使用SSD检测模型对图片进行目标检测，检测结果如下图所示
$ wget --no-check-certificate https://paddlehub.bj.bcebos.com/resources/test_img_bird.jpg
$ hub run ssd_mobilenet_v1_pascal --input_path test_img_bird.jpg
```
![SSD检测结果](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v0.5.0/docs/imgs/test_img_bird_output.jpg)

想了解更多PaddleHub已经发布的模型，请使用`hub search`命令查看所有已发布的模型。

```shell
$ hub search
```

## 深入了解PaddleHub
* [PaddleHub 介绍](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/home.md)
* [命令行工具](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/cmd_tool.md)
* [Finetune API与迁移学习](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/paddlehub_tl.md)
* [API](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/api/finetune_api.md)

## 答疑

当安装或者使用遇到问题时，可以通过[FAQ](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-FAQ)查找解决方案。
如果在FAQ中没有找到解决方案，欢迎您将问题和bug报告以[Github Issues](https://github.com/PaddlePaddle/PaddleHub/issues)的形式提交

## 版权和许可证
PaddleHub由[Apache-2.0 license](LICENSE)提供
