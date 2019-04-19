# senta

## 关于

本示例展示如何使用senta Module进行预测。

senta是中文情感分析模型，可以用于进行中文句子的情感分析，输出结果为`{正向/中性/负向}`中的一个，关于模型的训练细节，请查看[senta](https://github.com/baidu/senta)

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

`infer.sh`给出了使用命令行调用Module预测的示例脚本
通过以下命令试验下效果

```shell
$ sh infer.sh
```

## 通过python API预测

`infer_by_code.py`给出了使用python API调用Module预测的示例代码
通过以下命令试验下效果

```shell
python infer_by_code.py
```
