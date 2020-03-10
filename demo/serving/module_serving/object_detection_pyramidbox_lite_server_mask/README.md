# 部署图像分类服务-以pyramidbox_lite_server_mask为例
## 简介
目标检测作为深度学习常见任务，在各种场景下都有所使用。`pyramidbox_lite_server_mask`模型可以应用于口罩检测任务，关于`pyramidbox_lite_server_mask`的具体信息请参见[pyramidbox_lite_server_mask](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server_mask&en_category=ObjectDetection)。

使用PaddleHub Serving可以轻松部署一个在线目标检测服务API，可快速实现在线目标检测等web服务，如PaddlePaddle官网提供的[口罩检测场景示例](https://paddlepaddle.org.cn/hub/scene/maskdetect)，如下图所示：

<p align="center">  

<img src="../../../../docs/imgs/mask_demo.jpg" width="65%" />  

</p>

下面就带领大家使用PaddleHub Serving，通过简单几步部署一个目标检测服务。

## Step1：启动PaddleHub Serving
启动命令如下：
```shell
$ hub serving start -m pyramidbox_lite_server_mask
```
启动时会显示加载模型过程，启动成功后显示：
```shell
Loading pyramidbox_lite_server_mask successful.
```
这样就完成了一个口罩检测服务化API的部署，默认端口号为8866。

## Step2：测试图像生成在线API
我们用来测试的样例图片为：  

<p align="center">  

<img src="../../../../docs/imgs/family_mask.jpg" width="65%" />  

</p>  

<p align="center">  

<img src="../../../../docs/imgs/woman_mask.jpg" width="65%" />  

</p>

准备的数据格式为：
```python
files = [("image", file_1), ("image", file_2)]
```
**NOTE:** 文件列表每个元素第一个参数为"image"。

代码如下：
```python
>>> # 指定要检测的图片并生成列表[("image", img_1), ("image", img_2), ... ]
>>> file_list = ["../../../../docs/imgs/family_mask.jpg", "../../../../docs/imgs/girl_mask.jpg"]
>>> files = [("image", (open(item, "rb"))) for item in file_list]
```

## Step3：获取并验证结果
通过发送请求到目标检测服务API，就可得到结果，代码如下：
```python
>>> # 指定检测方法为pyramidbox_lite_server_mask并发送post请求
>>> url = "http://127.0.0.1:8866/predict/image/pyramidbox_lite_server_mask"
>>> r = requests.post(url=url, files=files)
```
我们可以打印接口返回结果：
```python
>>> results = eval(r.json()["results"])
>>> print(json.dumps(results, indent=4, ensure_ascii=False))
[
    {
        "data": [
            {
                "label": "MASK",
                "left": 455.5180733203888,
                "right": 658.8289226293564,
                "top": 186.38022020459175,
                "bottom": 442.67284870147705,
                "confidence": 0.92117363
            },
            {
                "label": "MASK",
                "left": 938.9076416492462,
                "right": 1121.0804233551025,
                "top": 326.9856423139572,
                "bottom": 586.0468536615372,
                "confidence": 0.997152
            },
            {
                "label": "NO MASK",
                "left": 1166.189564704895,
                "right": 1325.6211009025574,
                "top": 295.55220007896423,
                "bottom": 496.9406336545944,
                "confidence": 0.9346678
            }
        ],
        "path": "",
        "id": 1
    },
    {
        "data": [
            {
                "label": "MASK",
                "left": 1346.7342281341553,
                "right": 1593.7974529266357,
                "top": 239.36296990513802,
                "bottom": 574.6375751495361,
                "confidence": 0.95378655
            },
            {
                "label": "MASK",
                "left": 840.5126552581787,
                "right": 1083.8391423225403,
                "top": 417.5169044137001,
                "bottom": 733.8856244087219,
                "confidence": 0.85434145
            }
        ],
        "path": "",
        "id": 2
    }
]
```
根据结果可以看出准确识别了请求图片中的人脸位置及戴口罩确信度。

pyramidbox_lite_server_mask返回的结果还包括标注检测框的图像的base64编码格式，经过转换可以得到生成图像，代码如下：
```python
>>> for item in results:
...     with open(output_path, "wb") as fp:
...         fp.write(base64.b64decode(item["base64"].split(',')[-1]))
```
查看指定输出文件夹，就能看到生成图像了，如图：

<p align="center">  

<img src="./output/family_mask.jpg" width="65%" />  

</p>  

<p align="center">  

<img src="./output/woman_mask.jpg" width="65%" />  

</p>  


这样我们就完成了对目标检测服务化的部署和测试。

完整的测试代码见[pyramidbox_lite_server_mask_file_serving_demo.py](pyramidbox_lite_server_mask_file_serving_demo.py)。

## 进一步提升模型服务性能
`pyramidbox_lite_server_mask`还支持直接传入opencv mat表示的图片，不产生结果文件，而是直接输出检测的人脸位置和戴口罩概率，响应时间平均提升20%以上，可用于对响应时间和性能要求更高的场景。

使用直接传输数据的模式，仅需要修改上文Step2中的POST方法参数，具体如下：

```python
>>> with open(file="../../../../docs/imgs/family_mask.jpg", mode="rb") as fp:
...     base64_data = base64.b64encode(fp.read())
>>> base64_data = str(base64_data, encoding="utf8")
>>> data = {"b64s": [base64_data]}
>>> data = {"data": json.dumps(data)}
```
进行HTTP请求时只需将data参数传入即可，具体如下：
```python
>>> r = requests.post(url=url, data=data)
```
对结果的处理与上文一致，但需注意此种方法仅输出识别结果，不产生结果文件，因此不能获得生成图片。

完整的测试代码见[pyramidbox_lite_server_mask_serving_demo.py](pyramidbox_lite_server_mask_serving_demo.py)。


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
