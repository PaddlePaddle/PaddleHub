## 模型概述
AnimeGAN V2 图像风格转换模型

模型可将输入的图像转换成Shinkai风格

模型权重转换自AnimeGAN V2官方开源项目

模型所使用的权重为Shinkai-53.ckpt

模型详情请参考[AnimeGAN V2 开源项目](https://github.com/TachibanaYoshino/AnimeGANv2)

## 模型安装

```shell
$hub install animegan_v2_shinkai_53
```


## API 说明

```python
def style_transfer(
    self,
    images=None,
    paths=None,
    output_dir='output',
    visualization=False,
    min_size=32,
    max_size=1024
)
```

风格转换API，将输入的图片转换为漫画风格。

转换效果图如下：

![输入图像](https://ai-studio-static-online.cdn.bcebos.com/bd002c4bb6a7427daf26988770bb18648b7d8d2bfd6746bfb9a429db4867727f)
![输出图像](https://ai-studio-static-online.cdn.bcebos.com/fa4ba157e73c48658c4c9c6b8b92f5c99231d1d19556472788b1e5dd58d5d6cc)


**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，默认为 None；
* paths (list\[str\]): 图片的路径，默认为 None；
* visualization (bool): 是否将识别结果保存为图片文件，默认设为 False；
* output\_dir (str): 图片的保存路径，默认设为 output；
* min\_size (int): 输入图片的短边最小尺寸，默认设为 32；
* max\_size (int): 输入图片的短边最大尺寸，默认设为 1024。


**返回**

* res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]。


## 预测代码示例

```python
import cv2
import paddlehub as hub

# 模型加载
# use_gpu：是否使用GPU进行预测
model = hub.Module(name='animegan_v2_shinkai_53', use_gpu=False)

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
$ hub serving start -m animegan_v2_shinkai_53
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
url = "http://127.0.0.1:8866/predict/animegan_v2_shinkai_53"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```


## 模型相关信息

### 模型代码

https://github.com/TachibanaYoshino/AnimeGANv2

### 依赖

paddlepaddle >= 1.8.0

paddlehub >= 1.8.0
