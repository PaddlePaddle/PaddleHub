# 常见问题

## 使用`pip install paddlehub`时提示`Could not find a version that satisfies the requirement paddlehub (from versions: )`

这可能是因为pip指向了一个pypi的镜像源，该镜像源没有及时同步paddlehub版本导致。

使用如下命令来安装：

```shell
$ pip install -i https://pypi.org/simple/ paddlehub
```

## 使用paddlehub时，提示`ModuleNotFoundError: No module named 'paddle'`

这是因为PaddleHub依赖于PaddlePaddle，用户需要自行安装合适的PaddlePaddle版本。
如果机器不支持GPU，那么使用如下命令来安装PaddlePaddle的CPU版本：
```shell
$ pip install paddlepaddle
```

如果机器支持GPU，则使用如下命令来安装PaddlePaddle的GPU版本：
```shell
$ pip install paddlepaddle-gpu
```

## 利用PaddleHub ernie/bert进行Finetune时，提示`paddle.fluid.core_avx.EnforceNotMet: Input ShapeTensor cannot be found in Op reshape2`等信息

这是因为ernie/bert module的创建时和此时运行环境中PaddlePaddle版本不对应。
首先将PaddleHub升级至最新版本，同时将ernie卸载。
```shell
$ pip install --upgrade paddlehub
$ hub uninstall ernie
```

## 使用paddlehub时，无法下载预置数据集、module的等现象

下载数据集、module等，PaddleHub要求机器可以访问外网。可以使用server_check()可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully.
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully.
```

## PaddleHub Module是否支持多线程，如何加快Module训练或预测的速度。

PaddleHub Module不支持多线程，可以通过使用GPU、加大batch_size等措施加速训练或者预测。
以下示例为加速LAC Module分词的方法：

```python
results = lac.lexical_analysis(data=inputs, use_gpu=True, batch_size=10)
```

## 如何获取输入句子经过ERNIE编码后的句子表示Embedding？

具体参考[BERT Services]()使用说明

## 在虚拟机Python2环境中使用hub命令报错“Illegal instruction”

请先排除numpy的版本问题：使用pip show numpy检查numpy的版本，如果是1.16.0~1.16.3会出现[numpy错误](https://github.com/numpy/numpy/issues/9532)，请pip uinstall numpy后再用pip install numpy<1.17.0安装新版本的numpy。

## 如何修改PaddleHub的修改预训练模型存放路径？

通过设置系统环境变量HUB_HOME，修改预训练模型存放路径
