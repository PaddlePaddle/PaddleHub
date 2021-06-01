## 模型概述
UGATIT 图像风格转换模型

模型可将输入的人脸图像转换成动漫风格

模型权重来自UGATIT-Paddle开源项目

模型所使用的权重为genA2B_0835000

模型详情请参考[UGATIT-Paddle开源项目](https://github.com/miraiwk/UGATIT-paddle)

## 模型安装

```shell
$hub install UGATIT_83w
```


## API 说明

```python
def style_transfer(
    self,
    images=None,
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False
)
```

风格转换API，将输入的人脸图像转换成动漫风格。

转换效果图如下：

![输入图像](https://ai-studio-static-online.cdn.bcebos.com/d130fabd8bd34e53b2f942b3766eb6bbd3c19c0676d04abfbd5cc4b83b66f8b6)
![输出图像](https://ai-studio-static-online.cdn.bcebos.com/78653331ee2d472b81ff5bbccd6a904a80d2c5208f9c42c789b4f09a1ef46332)

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，默认为 None；
* paths (list\[str\]): 图片的路径，默认为 None；
* batch\_size (int): batch 的大小，默认设为 1；
* visualization (bool): 是否将识别结果保存为图片文件，默认设为 False；
* output\_dir (str): 图片的保存路径，默认设为 output。


**返回**

* res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]。


## 预测代码示例

```python
import cv2
import paddlehub as hub

# 模型加载
# use_gpu：是否使用GPU进行预测
model = hub.Module('UGATIT_83w', use_gpu=False)

# 模型预测
result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])

# or
# result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线图像风格转换服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m UGATIT_w83
```

这样就完成了一个图像风格转换的在线服务API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/UGATIT_w83"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```


## 模型相关信息

### 模型代码

https://github.com/miraiwk/UGATIT-paddle

### 依赖

paddlepaddle >= 1.8.0

paddlehub >= 1.8.0
