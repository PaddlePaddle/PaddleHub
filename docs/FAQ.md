# 安装相关问题

## 使用`pip install paddlehub`时提示`Could not find a version that satisfies the requirement paddlehub (from versions: )`

这可能是因为pip指向了一个pypi的镜像源，该镜像源没有及时同步paddlehub版本导致

使用如下命令来安装

```shell
$ pip install -i https://pypi.org/simple/ paddlehub
```

# 使用相关问题

## 使用paddlehub时，提示`ModuleNotFoundError: No module named 'paddle'`

这是因为PaddleHub依赖于PaddlePaddle，用户需要自行安装合适的PaddlePaddle版本

如果机器不支持GPU，那么使用如下命令来安装PaddlePaddle的CPU版本
```shell
$ pip install paddlepaddle
```

如果机器支持GPU，则使用如下命令来安装PaddlePaddle的GPU版本
```shell
$ pip install paddlepaddle-gpu
```
