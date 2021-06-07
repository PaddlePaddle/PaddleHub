## 命令行预测

```
hub run face_landmark_localization --input_path "/PATH/TO/IMAGE"
```

## API

```python
def __init__(face_detector_module=None)
```

**参数**

* face\_detector\_module (class): 人脸检测模型，默认为 ultra\_light\_fast\_generic\_face\_detector\_1mb\_640.


```python
def keypoint_detection(images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False,
                       output_dir='face_landmark_output',
                       visualization=False)
```

识别输入图片中的所有人脸关键点，每张人脸检测出68个关键点（人脸轮廓17个点，左右眉毛各5个点，左右眼睛各6个点，鼻子9个点，嘴巴20个点）

<p align="center">
<img src="https://paddlehub.bj.bcebos.com/resources/face_landmark.jpg"  hspace='5' width=500/> <br />
</p>

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，当为 None 时，默认设为face\_landmark\_output。

**返回**

* res (list\[dict\]): 识别结果的列表，列表元素为 dict, 有以下两个字段：
    * save\_path : 可视化图片的保存路径（仅当visualization=True时存在）；
    * data: 图片中每张人脸的关键点坐标


```python
def set_face_detector_module(face_detector_module)
```

设置为人脸关键点检测模型进行人脸检测的底座模型

**参数**

* face\_detector\_module (class): 人脸检测模型


```python
def get_face_detector_module()
```

获取为人脸关键点检测模型进行人脸检测的底座模型

**返回**

* 当前模型使用的人脸检测模型。

```python
def save_inference_model(dirname,
                         model_filename=None,
                         params_filename=None,
                         combined=True)
```

将模型保存到指定路径，由于人脸关键点检测模型由人脸检测+关键点检测两个模型组成，因此保存后会存在两个子目录，其中`face_landmark`为人脸关键点模型，`detector`为人脸检测模型。

**参数**

* dirname: 存在模型的目录名称
* model_filename: 模型文件名称，默认为\_\_model\_\_
* params_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
* combined: 是否将参数保存到统一的一个文件中


## 代码示例

```python
import paddlehub as hub
import cv2

face_landmark = hub.Module(name="face_landmark_localization")

# Replace face detection module to speed up predictions but reduce performance
# face_landmark.set_face_detector_module(hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320"))

result = face_landmark.keypoint_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = face_landmark.keypoint_detection(paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线人脸关键点检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m face_landmark_localization
```

这样就完成了一个人脸关键点服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64
import paddlehub as hub

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/face_landmark_localization"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

### 查看代码

https://github.com/lsy17096535/face-landmark

## Module贡献者

[Jason](https://github.com/jiangjiajun)

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
