# PaddleHub Senta

## 关于

本示例展示如何使用PaddleHub Senta Module进行预测。

Senta是百度NLP开放的中文情感分析模型，可以用于进行中文句子的情感分析，输出结果为`{正向/中性/负向}`中的一个，关于模型的结构细节，请查看[Senta](https://github.com/baidu/senta), 本示例代码选择的是Senta-BiLSTM模型。

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

## 命令行方式预测

`cli_demo.sh`给出了使用命令行接口 (Command Line Interface) 调用Module预测的示例脚本
通过以下命令体验下效果

```shell
$ sh cli_demo.sh
```

## 通过python API预测

`senta_demo.py`给出了使用python API调用Module预测的示例代码
通过以下命令试验下效果

```shell
python senta_demo.py
```

## 通过PaddleHub Finetune API微调
`senta_finetune.py` 给出了如何使用Senta模型的句子特征进行Fine-tuning的实例代码。
可以运行以下命令在ChnSentiCorp数据集上进行Fine-tuning.
```shell
$ sh run_finetune.sh
```
