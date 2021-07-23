# Linux/Mac 安装



## 环境依赖


* 操作系统：Mac/Linux
* Python >= 3.6.2
* PaddlePaddle >= 2.0.0
```python
    # 安装gpu版本的PaddlePaddle
    pip install paddlepaddle-gpu -U

    # 或者安装cpu版本的paddlepaddle
    # pip install paddlepaddle -U
```

注1、Python环境的安装问题，可以参考
[【零基础windows安装并实现图像风格迁移】](./windows_quickstart.md)的前2步，方法类似

注2、安装PaddlePaddle深度学习框架，如果遇到问题，可查阅[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick)

## PaddleHub安装命令


```python
    pip install paddlehub==2.1.0
```
除上述依赖外，PaddleHub的预训练模型和预置数据集需要连接服务端进行下载，请确保机器可以正常访问网络。若本地已存在相关的数据集和预训练模型，则可以离线运行PaddleHub。

使用PaddleHub下载数据集、预训练模型等，要求机器可以访问外网。可以使用`server_check()`可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：


```python
    import paddlehub
    paddlehub.server_check()
    # 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully。
    # 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully。
```


## 极简代码测试
进入Python环境下，测试以下代码得到预期结果，则说明PaddleHub安装成功

```python
import paddlehub as hub

lac = hub.Module(name="lac")
test_text = ["今天是个好天气。"]

results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
print(results)
#{'word': ['今天', '是', '个', '好天气', '。'], 'tag': ['TIME', 'v', 'q', 'n', 'w']}
```
