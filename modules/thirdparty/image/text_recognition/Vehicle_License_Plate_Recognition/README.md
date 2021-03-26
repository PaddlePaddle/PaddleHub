## 概述
Vehicle_License_Plate_Recognition 是一个基于 CCPD 数据集训练的车牌识别模型，能够检测出图像中车牌位置并识别其中的车牌文字信息，大致的模型结构如下，分为检测车牌和文字识别两个模块：

![](https://ai-studio-static-online.cdn.bcebos.com/35a3dab32ac948549de41afba7b51a5770d3f872d60b437d891f359a5cef8052)

## API
```python
def plate_recognition(images)
```
车牌识别 API

**参数**
* images(str / ndarray / list(str) / list(ndarray))：待识别图像的路径或者图像的 Ndarray(RGB)

**返回**
* results(list(dict{'license', 'bbox'})): 识别到的车牌信息列表，包含车牌的位置坐标和车牌号码

**代码示例**
```python
import paddlehub as hub

# 加载模型
model = hub.Module(name='Vehicle_License_Plate_Recognition')

# 车牌识别
result = model.plate_recognition("test.jpg")

# 打印结果
print(result)
```
    [{'license': '苏B92912', 'bbox': [[131.0, 251.0], [368.0, 253.0], [367.0, 338.0], [131.0, 336.0]]}]

## 服务部署

PaddleHub Serving 可以部署一个在线车牌识别服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start --modules Vehicle_License_Plate_Recognition
```

这样就完成了一个车牌识别的在线服务API的部署，默认端口号为8866。

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
data = {'images':[cv2_to_base64(cv2.imread("test.jpg"))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/Vehicle_License_Plate_Recognition"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```
    [{'bbox': [[260.0, 100.0], [546.0, 104.0], [544.0, 200.0], [259.0, 196.0]], 'license': '苏DS0000'}]

## 查看代码
https://github.com/jm12138/License_plate_recognition

## 依赖
paddlepaddle >= 2.0.0

paddlehub >= 2.0.4

paddleocr >= 2.0.2