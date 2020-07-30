# PaddleHub安装
## 环境准备
PaddleHub需要与飞桨一起使用，其硬件和操作系统的适用范围与[飞桨](https://www.paddlepaddle.org.cn/install/quick)相同。
> 注意：飞桨版本需要>= 1.7.0。  


```python
# 查看是否安装飞桨
$python # 进入python解释器
import paddle.fluid
paddle.fluid.install_check.run_check()
```

> 如果出现`Your Paddle Fluid is installed successfully`，说明飞桨已成功安装。


```python
$pip list | grep paddlepaddle # 查看飞桨版本。pip list查看所有的package版本，grep负责根据关键字筛选。
```

## 安装操作
根据实际需要，执行以下命令之一进行PaddleHub的安装（推荐使用第一个）。

> 1.安装过程中需要网络连接，请确保机器可以正常访问网络。成功安装之后，可以离线使用。  
2.如果已安装PaddleHub，再次执行安装操作将先卸载再安装。安装方式支持：安装指定版本和安装最新版本。  
3.由于国内网速的问题，直接pip安装包通常速度非常慢，而且经常会出现装到一半失败了的问题。使用国内镜像可以节省时间，提高pip安装的效率。  
  ```
  国内镜像源列表：  
  清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/  
  百度：https://mirror.baidu.com/pypi/simple
  ```


```python
$pip install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple  # 安装最新版本，使用清华源
```


```python
$pip install paddlehub==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple # 安装指定版本（==1.6.1表示PaddleHub的版本），使用清华源
```


```python
$pip install paddlehub --upgrade -i https://mirror.baidu.com/pypi/simple  # 安装最新版本，使用百度源
```


```python
$pip install paddlehub==1.6.1 -i https://mirror.baidu.com/pypi/simple # 安装指定版本（==1.6.1表示PaddleHub的版本），使用百度源
```

> 如果出现`Successfully installed paddlehub`，说明PaddleHub安装成功。

## 验证安装
检查PaddleHub是否安装成功。


```python
$pip list | grep paddlehub # pip list查看所有的package版本，grep负责根据关键字筛选
```


```python
$pip show paddlehub # 查看PaddleHub详细信息
```

PaddleHub详细信息的如下面所示，可以查看显示了PaddleHub的版本、位置等信息。
```
Name: paddlehub
Version: 1.7.1
Summary: A toolkit for managing pretrained models of PaddlePaddle and helping user getting started with transfer learning more efficiently.
Home-page: https://github.com/PaddlePaddle/PaddleHub
Author: PaddlePaddle Author
Author-email: paddle-dev@baidu.com
License: Apache 2.0
Location: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages
Requires: pandas, pre-commit, gunicorn, flake8, visualdl, yapf, flask, protobuf, sentencepiece, six, cma, colorlog
Required-by:
```

## 如何卸载
此卸载仅卸载PaddleHub，已下载的模型文件和数据集仍保留。


```python
$pip uninstall paddlehub -y  # 卸载PaddleHub
```

> 如果出现`Successfully uninstalled paddlehub`,表明PaddleHub卸载成功。

## 常见问题
1. 已安装PaddleHub，可以升级飞桨版本吗？  
 	答复：可以。直接正常升级飞桨版本即可。  
2. 已安装PaddleHub，如何升级？  
	答复：执行`pip install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple`，可以将PaddleHub升级到最新版本。  
3. `upgrade`安装与安装指定版本有什么区别？  
 	答复：`upgrade`安装的是最新版本，安装指定版本可以安装任意版本。  
4. 如何设置PaddleHub下载的缓存位置？  
 	答复：PaddleHub的Module默认会保存在用户目录下，可以通过修改环境变量`HUB_HOME`更改这个位置。
