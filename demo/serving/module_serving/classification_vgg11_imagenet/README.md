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
首先引入需要的包：
```python
>>> import requests
>>> import json
```

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
>>> results = eval(r.json()["results"])
>>> print(json.dumps(results, indent=4, ensure_ascii=False))
[
    [
        {
            "Egyptian cat": 0.540287435054779
        }
    ],
    [
        {
            "daisy": 0.9976677298545837
        }
    ]
]
```

这样我们就完成了对图像分类预测服务化部署和测试。

完整的测试代码见[vgg11_imagenent_serving_demo.py](vgg11_imagenet_serving_demo.py)。
