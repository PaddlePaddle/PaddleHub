## 模型概述
openpose 手部关键点检测模型

模型详情请参考[openpose开源项目](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## 模型安装

```shell
$hub install hand_pose_localization
```

## API 说明

```python
def keypoint_detection(
    self,
    images=None,
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False
)
```

预测API，识别出人体手部关键点。

![手部关键点](https://ai-studio-static-online.cdn.bcebos.com/97e1ae7c1e68477d85b37f53ee997fbc4ef0fc12c7634301bc08749bd003cac0)

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\], 默认设为 None；
* paths (list\[str\]): 图片的路径, 默认设为 None；
* batch\_size (int): batch 的大小，默认设为 1；
* visualization (bool): 是否将识别结果保存为图片文件，默认设为 False；
* output\_dir (str): 图片的保存路径，默认设为 output。

**返回**

* res (list[list[list[int]]]): 每张图片识别到的21个手部关键点组成的列表，每个关键点的格式为[x, y]，若有关键点未识别到则为None


## 预测代码示例

```python
import cv2
import paddlehub as hub

# use_gpu：是否使用GPU进行预测
model = hub.Module('hand_pose_localization', use_gpu=False)

# 调用关键点检测API
result = model.keypoint_detection(images=[cv2.imread('/PATH/TO/IMAGE')])

# or
# result = model.keypoint_detection(paths=['/PATH/TO/IMAGE'])

# 打印预测结果
print(result)
```

## 服务部署

PaddleHub Serving可以部署一个在线人体手部关键点检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m hand_pose_localization
```

这样就完成了一个人体手部关键点检测的在线服务API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64

# 图片Base64编码函数
def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/hand_pose_localization"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```


## 模型相关信息

### 模型代码

https://github.com/CMU-Perceptual-Computing-Lab/openpose

### 依赖

paddlepaddle >= 1.8.0

paddlehub >= 1.8.0
