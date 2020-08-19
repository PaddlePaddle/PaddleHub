## 模型概述

DCSCN是基于Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network设计的轻量化超分辨模型。该模型使用残差结构和跳连的方式构建网络来提取局部和全局特征，同时使用并行1*1的卷积网络学习细节特征提升模型性能。该模型提供的超分倍数为2倍。

## 命令行预测

```
$ hub run dcscn --input_path "/PATH/TO/IMAGE"

```

## API

```python
def reconstruct(self,
                images=None,
                paths=None,
                use_gpu=False,
                visualization=False,
                output_dir="dcscn_output")
```

预测API，用于图像超分辨率。

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* use\_gpu (bool): 是否使用 GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径。

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 'save\_path', 'data'，对应的取值为：
  * save\_path (str, optional): 可视化图片的保存路径（仅当visualization=True时存在）；
  * data (numpy.ndarray): 超分辨后图像。

```python
def save_inference_model(self,
                         dirname='dcscn_save_model',
                         model_filename=None,
                         params_filename=None,
                         combined=False)
```

将模型保存到指定路径。

**参数**

* dirname: 存在模型的目录名称
* model\_filename: 模型文件名称，默认为\_\_model\_\_
* params\_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
* combined: 是否将参数保存到统一的一个文件中

## 代码示例

```python
import cv2
import paddlehub as hub

sr_model = hub.Module('dcscn')
im = cv2.imread('/PATH/TO/IMAGE').astype('float32')
#visualization=True可以用于查看超分图片效果，可设置为False提升运行速度。
res = sr_model.reconstruct(images=[im], visualization=True)
print(res[0]['data'])
sr_model.save_inference_model()
```

## 服务部署

PaddleHub Serving可以部署一个图像超分的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m dcscn
```

这样就完成了一个超分任务的服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

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

org_im = cv2.imread('/PATH/TO/IMAGE')
data = {'images':[cv2_to_base64(org_im)]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/dcscn"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

sr = np.expand_dims(cv2.cvtColor(base64_to_cv2(r.json()["results"][0]['data']), cv2.COLOR_BGR2GRAY), axis=2)
shape =sr.shape
org_im = cv2.cvtColor(org_im, cv2.COLOR_BGR2YUV)
uv = cv2.resize(org_im[...,1:], (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
combine_im =  cv2.cvtColor(np.concatenate((sr, uv), axis=2), cv2.COLOR_YUV2BGR)
cv2.imwrite('dcscn_X2.png', combine_im)
print("save image as dcscn_X2.png")

```
### 查看代码

https://github.com/jiny2001/dcscn-super-resolution



### 依赖

paddlepaddle >= 1.8.0

paddlehub >= 1.7.1
