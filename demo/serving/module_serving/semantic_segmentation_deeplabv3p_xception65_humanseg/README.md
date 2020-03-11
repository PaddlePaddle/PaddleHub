# 部署图像分割服务-以deeplabv3p_xception65_humanseg为例
## 简介
图像分割是深度学习的常见任务。使用`deeplabv3p_xception65_humanseg`模型可以进行人像分割任务，关于`deeplabv3p_xception65_humanseg`的具体信息请参见[deeplabv3p_xception65_humanseg](https://paddlepaddle.org.cn/hubdetail?name=deeplabv3p_xception65_humanseg&en_category=ImageSegmentation)。

使用PaddleHub Serving可以轻松部署一个在线图像分割服务API，可将此API接入自己的web网站进行在线图像分割，也可接入移动端应用程序，实现拍照分割等功能。

下面就带领大家使用PaddleHub Serving，通过简单几步部署一个目标检测服务。

## Step1：启动PaddleHub Serving
启动命令如下
```shell
$ hub serving start -m deeplabv3p_xception65_humanseg
```
启动时会显示加载模型过程，启动成功后显示：
```shell
Loading deeplabv3p_xception65_humanseg successful.
```
这样就完成了一个图像分割服务化API的部署，默认端口号为8866。

## Step2：测试图像分割在线API
我们用来测试的样例图片为：  

<p align="center">  

<img src="../img/girl.jpg" width="65%" />  

</p>  

准备的数据格式为：
```python
files = [("image", file_1), ("image", file_2)]
```
**NOTE:** 文件列表每个元素第一个参数为"image"。

代码如下
```python
>>> # 指定要检测的图片并生成列表[("image", img_1), ("image", img_2), ... ]
>>> file_list = ["../../../../docs/imgs/girl.jpg"]
>>> files = [("image", (open(item, "rb"))) for item in file_list]
```

## Step3：获取并验证结果
然后就可以发送请求到图像分割服务API，并得到结果，代码如下：
```python
>>> # 指定检测方法为deeplabv3p_xception65_humanseg并发送post请求
>>> url = "http://127.0.0.1:8866/predict/image/deeplabv3p_xception65_humanseg"
>>> r = requests.post(url=url, files=files)
```
我们可以打印接口返回结果：
```python
>>> results = eval(r.json()["results"])
>>> print(json.dumps(results, indent=4, ensure_ascii=False))
{
    "status": "0",
    "msg": "",
    "results": [
        {
            "origin": "girl.jpg",
            "processed": "humanseg_output/girl.png"
        }
    ]
}
```

deeplabv3p_xception65_humanseg返回的结果还包括人像分割后的图像的base64编码格式，经过转换可以得到生成图像，代码如下：
```python
>>> for item in results:
...     with open(output_path, "wb") as fp:
...         fp.write(base64.b64decode(item["base64"].split(',')[-1]))
```
查看指定输出文件夹，就能看到生成图像了，如图：

<p align="center">  

<img src="./output/girl.png" width="65%" />  

</p>  

这样我们就完成了对图像分割模型deeplabv3p_xception65_humanseg服务化的部署和测试。

完整的测试代码见[deeplabv3p_xception65_humanseg_serving_demo.py](deeplabv3p_xception65_humanseg_serving_demo.py)。

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

此Demo的具体信息和代码请参见[LAC Serving_2.1.0](../lexical_analysis_lac/lac_2.1.0_serving_demo.py)。
