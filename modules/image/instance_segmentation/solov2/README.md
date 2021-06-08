## 模型概述

solov2是基于'SOLOv2: Dynamic, Faster and Stronger'实现的快速实例分割的模型。该模型基于SOLOV1, 并且针对mask的检测效果和运行效率进行改进，在实例分割任务中表现优秀。相对语义分割，实例分割需要标注出图上同一物体的不同个体。solov2实例分割效果如下：

<div align="center">
<img src="example.png"  width = "642" height = "400" />
</div>


## API

```python
def predict(self,
            image: Union[str, np.ndarray],
            threshold: float = 0.5,
            visualization: bool = False,
            save_dir: str = 'solov2_result'):
```

预测API，实例分割。

**参数**

* image (Union\[str, np.ndarray\]): 图片路径或者图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* threshold (float): 检测模型输出结果中，预测得分低于该阈值的框将被滤除，默认值为0.5；
* visualization (bool): 是否将可视化图片保存；
* save_dir (str): 保存图片到路径， 默认为"solov2_result"。

**返回**

* res (dict): 识别结果，关键字有 'segm', 'label', 'score'对应的取值为：
  * segm (np.ndarray): 实例分割结果,取值为0或1。0表示背景，1为实例；
  * label (list): 实例分割结果类别id；
  * score (list):实例分割结果类别得分；


## 代码示例

```python
import cv2
import paddlehub as hub

img = cv2.imread('/PATH/TO/IMAGE')
model = hub.Module(name='solov2', use_gpu=False)
output = model.predict(image=img,visualization=False)
```

## 服务部署

PaddleHub Serving可以部署一个实例分割的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m solov2
```

默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64

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

org_im = cv2.imread('/PATH/TO/IMAGE')
h, w, c = org_im.shape
data = {'images':[cv2_to_base64(org_im)]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/solov2"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

seg = base64.b64decode(r.json()["results"]['segm'].encode('utf8'))
seg = np.fromstring(seg, dtype=np.int32).reshape((-1, h, w))

label = base64.b64decode(r.json()["results"]['label'].encode('utf8'))
label = np.fromstring(label, dtype=np.int64)

score = base64.b64decode(r.json()["results"]['score'].encode('utf8'))
score = np.fromstring(score, dtype=np.float32)

print('seg', seg)
print('label', label)
print('score', score)
```

### 查看代码

https://github.com/PaddlePaddle/PaddleDetection


### 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.0.0
