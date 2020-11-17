## 模型概述

photo_restoration 是针对老照片修复的模型。它主要由两个部分组成：着色和超分。着色模型基于
https://github.com/jantic/DeOldify
，超分模型基于Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Mode. 用户可以根据自己的需求选择对图像进行着色或超分操作。
因此在使用该模型时，请预先安装deoldify和realsr两个模型。


## API

```python
def run_image(self,
              input,
              model_select= ['Colorization', 'SuperResolution'],
              save_path = 'photo_restoration'): 
```

预测API，用于图片修复。

**参数**

* input (numpy.ndarray｜str): 图片数据，numpy.ndarray 或者 str形式。ndarray.shape 为 \[H, W, C\]，BGR格式; str为图片的路径。

* model_select (list\[str\]): 选择对图片对操作，\['Colorization'\]对图像只进行着色操作， \['SuperResolution'\]对图像只进行超分操作；
默认值为\['Colorization', 'SuperResolution'\]。

* save_path (str): 保存图片的路径, 默认为'photo_restoration'。

**返回**

* output (numpy.ndarray): 照片修复结果，ndarray.shape 为 \[H, W, C\]，BGR格式。



## 代码示例

图片修复代码示例：

```python
import cv2
import paddlehub as hub

model = hub.Module('photo_restoration', visualization=True)
im = cv2.imread('/PATH/TO/IMAGE')
res = model.run_image(im)

```

## 服务部署

PaddleHub Serving可以部署一个人像分割的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m photo_restoration
```

这样就完成了一个人像分割的服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import base64

import cv2
import numpy as np

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')
def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

# 发送HTTP请求
org_im = cv2.imread('PATH/TO/IMAGE')
data = {'images':cv2_to_base64(org_im), 'model_select': ['Colorization', 'SuperResolution']}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/photo_restoration"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
img = base64_to_cv2(r.json()["results"])
cv2.imwrite('PATH/TO/SAVE/IMAGE', img)
```

### 依赖

paddlepaddle >= 2.0.0rc

paddlehub >= 1.8.2
