# Bert Service  
## 简介
### 什么是Bert Service
`Bert Service`是基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)框架的快速部署模型远程计算服务方案，可将embedding过程通过调用API接口的方式实现，减少了对机器资源的依赖。使用PaddleHub可在服务器上一键部署`Bert Service`服务，在另外的普通机器上通过客户端接口即可轻松的获取文本对应的embedding数据。  

**NOTE:** 关于`Bert Service`的更多信息请参阅[Bert Serving](../../../tutorial/bert_service.md)。

## Demo-利用Bert Service部署ernie_tiny在线embedding服务
在这里，我们将展示一个实际场景中可能使用的demo，我们利用PaddleHub在一台GPU机器上部署`ernie_tiny`模型服务，并在另一台CPU机器上尝试访问，获取一首七言绝句的embedding。
### Step1：安装环境依赖
首先需要安装环境依赖，根据第2节内容分别在两台机器上安装相应依赖。  

### Step2：启动Bert Service服务端
确保环境依赖安装正确后，在要部署服务的GPU机器上使用PaddleHub命令行工具启动`Bert Service`服务端，命令如下：
```shell
$ hub serving start bert_service -m ernie_tiny --use_gpu --gpu 0 --port 8866
```
启动成功后打印
```shell
Server[baidu::paddle_serving::predictor::bert_service::BertServiceImpl] is serving on port=8866.
```  
这样就启动了`ernie_tiny`的在线服务，监听8866端口，并在0号GPU上进行任务。
### Step3：使用Bert Service客户端进行远程调用  
部署好服务端后，就可以用普通机器作为客户端测试在线embedding功能。

首先导入客户端依赖。  
```python
from paddlehub.serving.bert_serving import bs_client
```

接着启动并初始化`bert service`客户端`BSClient`(这里的server为虚拟地址，需根据自己实际ip设置)
```python
bc = bs_client.BSClient(module_name="ernie_tiny", server="127.0.0.1:8866")
```

然后输入文本信息。
```python
input_text = [["西风吹老洞庭波"], ["一夜湘君白发多"], ["醉后不知天在水"], ["满船清梦压星河"], ]
```

最后利用客户端接口`get_result`发送文本到服务端，以获取embedding结果。
```python
result = bc.get_result(input_text=input_text)
```
最后即可得到embedding结果(此处只展示部分结果)。
```python
[[0.9993321895599361, 0.9994612336158751, 0.9999646544456481, 0.732795298099517, -0.34387934207916204, ... ]]
```
客户端代码demo文件见[示例](../paddlehub/serving/bert_serving/bert_service.py)。  
运行命令如下：  
```shell
$ python bert_service_client.py
```  

运行过程如下图：

<div align="center">  

&emsp;&emsp;<img src="https://github.com/ShenYuhan/ml-python/blob/master/short_client_fast.gif" aligh="center" width="70%" alt="启动BS" />  

</div>  

### Step4：关闭Bert Service服务端
如要停止`Bert Service`服务端程序，可在其启动命令行页面使用Ctrl+C方式关闭，关闭成功会打印如下日志：
```shell
Paddle Inference Server exit successfully!
```
这样，我们就利用一台GPU机器就完成了`Bert Service`的部署，并利用另一台普通机器进行了测试，可见通过`Bert Service`能够方便地进行在线embedding服务的快速部署。  
