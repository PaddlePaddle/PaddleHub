## 模型概述

HUmanSeg_lite是在ShuffleNetV2网络结构的基础上进行优化，进一步减小了网络规模，网络大小只有541K，量化后只有187K，适用于手机自拍人像分割，且能在移动端进行实时分割。


## 命令行预测

```
hub run humanseg_lite --input_path "/PATH/TO/IMAGE"

```


## API

```python
def segment(images=None,
            paths=None,
            batch_size=1,
            use_gpu=False,
            visualization=True,
            output_dir='humanseg_output')
```

预测API，用于人像分割。

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径。

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 'save\_path', 'data'，对应的取值为：
  * save\_path (str, optional): 可视化图片的保存路径（仅当visualization=True时存在）；
  * data (numpy.ndarray): 人像分割结果，仅包含Alpha通道，取值为0-255 (0为全透明，255为不透明)，也即取值越大的像素点越可能为人体，取值越小的像素点越可能为背景。

```python
def save_inference_model(dirname,
                         model_filename=None,
                         params_filename=None,
                         combined=True)
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

human_seg = hub.Module('humanseg_lite')
im = cv2.imread('/PATH/TO/IMAGE')
res = human_seg.segment(images=[im],visualization=True)
```

## 服务部署

PaddleHub Serving可以部署一个人像分割的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m humanseg_lite
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
data = {'images':[cv2_to_base64(cv2.imread('/PATH/TO/IMAGE'))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/humanseg_lite"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(base64_to_cv2(r.json()["results"][0]['data']))
```




### 依赖

paddlepaddle >= 1.8.1

paddlehub >= 1.7.1
