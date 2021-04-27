# 常见问题

## 使用pip install paddlehub时提示
`Could not find a version that satisfies the requirement paddlehub (from versions: )`

这可能是因为pip指向了一个pypi的镜像源，该镜像源没有及时同步paddlehub版本导致。

使用如下命令来安装：

```shell
$ pip install -i https://pypi.org/simple/ paddlehub
```

## 使用paddlehub时，提示
`ModuleNotFoundError: No module named 'paddle'`

这是因为PaddleHub依赖于PaddlePaddle，用户需要自行安装合适的PaddlePaddle版本。
如果机器不支持GPU，那么使用如下命令来安装PaddlePaddle的CPU版本：
```shell
$ pip install paddlepaddle
```

如果机器支持GPU，则使用如下命令来安装PaddlePaddle的GPU版本：
```shell
$ pip install paddlepaddle-gpu
```

## 使用paddlehub时，无法下载预置数据集、module的等现象

下载数据集、module等，PaddleHub要求机器可以访问外网。可以使用server_check()可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

```python
import paddlehub
paddlehub.server_check()
# 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully.
# 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully.
```

## PaddleHub Module是否支持多线程加速预测？

由于PaddlePaddle自身的限制，PaddleHub无法通过多线程来加速模型预测。

## 如何修改PaddleHub的修改预训练模型存放路径？

通过设置系统环境变量HUB_HOME，修改预训练模型存放路径
