# 部署图像分类服务-以vgg11_imagenent为例
## 1 简介
&emsp;&emsp;图像分类是指通过模型，预测给定的图片所属类别，vgg11_imagenent就是一种有效的图像分类模型。关于vgg11_imagenent的具体信息请参阅[vgg11_imagenent](https://paddlepaddle.org.cn/hubdetail?name=vgg11_imagenet&en_category=ImageClassification)。

&emsp;&emsp;使用PaddleHub-Serving可以部署一个在线图片分类服务，既可以对用户暴露直接预测接口，也可以利用此接口实现一个web网站，甚至可以集成到移动端应用程序中实现拍照识别功能。

&emsp;&emsp;这里就带领大家使用PaddleHub-Serving，通过简单几步部署一个图像分类服务。

## 2 启动PaddleHub-Serving
&emsp;&emsp;启动命令如下
```shell
$ hub serving start -m vgg11_imagenet  
```
&emsp;&emsp;启动时会显示加载模型过程，启动成功后显示
```shell
Loading vgg11_imagenet successful.
```
&emsp;&emsp;这样就完成了一个图像分类服务化API的部署，默认端口号为8866。

## 3 测试图像分类在线API
&emsp;&emsp;我们用来测试的样例图片为  

<p align="center">  
<img src="../img/cat.jpg" width="80%" />  
</p>  

<p align="center">  
<img src="../img/flower.jpg" width="80%"/>  
</p>


&emsp;&emsp;准备的数据格式为
```python
files = [("image", file_1), ("image", file_2)]
```
&emsp;&emsp;注意每个元素第一个参数为"image"。

&emsp;&emsp;代码如下
```python
>>> file_list = ["../img/cat.jpg", "../img/flower.jpg"]  
>>> files = [("image", (open(item, "rb"))) for item in file_list]
```
&emsp;&emsp;然后就可以发送请求到图像分类服务API，并得到结果了，代码如下
```python
>>> url = "http://127.0.0.1:8866/predict/image/vgg11_imagenet"
>>> r = requests.post(url=url, files=files)
```
&emsp;&emsp;vgg11_imagenent返回的结果为图像分类结果及其对应的概率，我们尝试打印接口返回结果
```python
>>> print(json.dumps(r.json(), indent=4, ensure_ascii=False))  
{
    "results": "[[{'Egyptian cat': 0.540287435054779}], [{'daisy': 0.9976677298545837}]]"
}
```
&emsp;&emsp;这样我们就完成了对图像分类预测服务化部署和测试。完整的测试代码见[vgg11_imagenent_serving_demo.py](./vgg11_imagenet_serving_demo.py)。
