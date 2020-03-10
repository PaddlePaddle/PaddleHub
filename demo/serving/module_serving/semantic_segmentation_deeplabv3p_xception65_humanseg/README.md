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
[
        {
            "origin": "girl.jpg",
            "processed": "humanseg_output/girl.png"
        }
]
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
