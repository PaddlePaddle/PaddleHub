# 安装

## 环境依赖

* Python >= 3.6
* PaddlePaddle >= 1.7.0
* 操作系统：Windows/Mac/Linux

## 安装命令

在安装PaddleHub之前，请先安装PaddlePaddle深度学习框架，更多安装说明请查阅[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick)

```shell
pip install paddlehub
```

除上述依赖外，PaddleHub的预训练模型和预置数据集需要连接服务端进行下载，请确保机器可以正常访问网络。若本地已存在相关的数据集和预训练模型，则可以离线运行PaddleHub。

**NOTE:** 使用PaddleHub下载数据集、预训练模型等，要求机器可以访问外网。可以使用`server_check()`可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully。
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully。
```
