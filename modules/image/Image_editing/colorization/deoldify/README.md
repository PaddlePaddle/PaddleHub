
## 模型概述
deoldify是用于图像和视频的着色渲染模型，该模型能够实现给黑白照片和视频恢复原彩。

## API 说明

```python
def predict(self, input):
```

着色变换API，得到着色后的图片或者视频。


**参数**

* input(str): 图片或者视频的路径；

**返回**

若输入是图片，返回值为：
* pred_img(np.ndarray): BGR图片数据；
* out_path(str): 保存图片路径。

若输入是视频，返回值为：
* frame_pattern_combined(str): 视频着色后单帧数据保存路径；
* vid_out_path(str): 视频保存路径。

```python
def run_image(self, img):
```
图像着色API， 得到着色后的图片。

**参数**

* img (str｜np.ndarray): 图片路径或则BGR格式图片。

**返回**

* pred_img(np.ndarray): BGR图片数据；

```python
def run_video(self, video):
```
视频着色API， 得到着色后的视频。

**参数**

* video (str): 待处理视频路径。

**返回**

* frame_pattern_combined(str): 视频着色后单帧数据保存路径；
* vid_out_path(str): 视频保存路径。

## 预测代码示例

```python
import paddlehub as hub

model = hub.Module(name='deoldify')
model.predict('/PATH/TO/IMAGE/OR/VIDEO')
```

## 服务部署

PaddleHub Serving可以部署一个在线照片着色服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m deoldify
```

这样就完成了一个图像着色的在线服务API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

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
org_im = cv2.imread('/PATH/TO/ORIGIN/IMAGE')
data = {'images':cv2_to_base64(org_im)}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/deoldify"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
img = base64_to_cv2(r.json()["results"])
cv2.imwrite('/PATH/TO/SAVE/IMAGE', img)
```


## 模型相关信息

### 模型代码

https://github.com/jantic/DeOldify

### 依赖

paddlepaddle >= 2.0.0rc

paddlehub >= 1.8.3
