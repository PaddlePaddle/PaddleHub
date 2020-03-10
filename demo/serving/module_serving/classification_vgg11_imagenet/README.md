# 部署图像分类服务-以vgg11_imagenent为例
## 简介
图像分类是指通过模型，预测给定的图片所属类别，vgg11_imagenent就是一种有效的图像分类模型。关于vgg11_imagenent的具体信息请参见[vgg11_imagenent](https://paddlepaddle.org.cn/hubdetail?name=vgg11_imagenet&en_category=ImageClassification)。

使用PaddleHub Serving可以部署一个在线图片分类服务，既可以对用户暴露直接预测接口，也可以利用此接口实现一个web网站，甚至可以集成到移动端应用程序中实现拍照识别功能。

这里就带领大家使用PaddleHub Serving，通过简单几步部署一个图像分类服务。

##  Step1：启动PaddleHub Serving
启动命令如下：
```shell
$ hub serving start -m vgg11_imagenet  
```
启动时会显示加载模型过程，启动成功后显示：
```shell
Loading vgg11_imagenet successful.
```
这样就完成了一个图像分类服务化API的部署，默认端口号为8866。

## Step2：测试图像分类在线API
我们用来测试的样例图片为：  

<p align="center">  
<img src="../../../../docs/imgs/cat.jpg" width="45%" />  
</p>  

<p align="center">  
<img src="../../../../docs/imgs/flower.jpg" width="45%"/>  
</p>

准备的数据格式为：
```python
files = [("image", file_1), ("image", file_2)]
```
**NOTE:** 每个元素第一个参数为"image"。

代码如下：
```python
>>> file_list = ["../img/cat.jpg", "../img/flower.jpg"]  
>>> files = [("image", (open(item, "rb"))) for item in file_list]
```

## Step3：获取并验证结果
然后就可以发送请求到图像分类服务API，并得到结果了，代码如下：
```python
>>> # 指定检测方法为vgg11_imagenet并发送post请求
>>> url = "http://127.0.0.1:8866/predict/image/vgg11_imagenet"
>>> r = requests.post(url=url, files=files)
```
vgg11_imagenent返回的结果为图像分类结果及其对应的概率，我们尝试打印接口返回结果：
```python

>>> print(json.dumps(r.json(), indent=4, ensure_ascii=False))  
{
    "results": "[[{'Egyptian cat': 0.540287435054779}], [{'daisy': 0.9976677298545837}]]"
}
```

这样我们就完成了对图像分类预测服务化部署和测试。

完整的测试代码见[vgg11_imagenent_serving_demo.py](vgg11_imagenet_serving_demo.py)。

## 客户端请求新版模型的方式
对某些新版模型，客户端请求方式有所变化，更接近本地预测的请求方式，以降低学习成本。
以lac(2.1.0)为例，使用上述方法进行请求将提示：
```python
{
    "Warnning": "This usage is out of date, please use 'application/json' as content-type to post to /predict/lac. See 'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details."
}
```
对于lac(2.1.0)，请求的方式如下：
```python
# coding: utf8
import requests
import json

if __name__ == "__main__":
    # 指定用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text = ["今天是个好日子", "天气预报说今天要下雨"]
    # 以key的方式指定text传入预测方法的时的参数，此例中为"texts"
    # 对应本地部署，则为lac.analysis_lexical(texts=[text1, text2])
    data = {"texts": text}
    # 指定预测方法为lac并发送post请求
    url = "http://127.0.0.1:8866/predict/lac"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

此Demo的具体信息和代码请参见[LAC Serving_2.1.0](../../demo/serving/module_serving/lexical_analysis_lac/lac_2.1.0_serving_demo.py)。
