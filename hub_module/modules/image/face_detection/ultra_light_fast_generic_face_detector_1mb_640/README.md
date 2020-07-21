## 命令行预测

```
hub run ultra_light_fast_generic_face_detector_1mb_640 --input_path "/PATH/TO/IMAGE"
```

## API

```python
def face_detection(images=None,
                   paths=None,
                   batch_size=1,
                   use_gpu=False,
                   visualization=False,
                   output_dir=None,
                   confs_threshold=0.5):
```

检测输入图片中的所有人脸位置

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，当为 None 时，默认设为face\_detector\_640\_predict\_output；
* confs\_threshold (float): 置信度的阈值。

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 path, save\_path, data，其中：
  * path 字段为原输入图片的路径（仅当使用paths输入时存在）；
  * save\_path 字段为可视化图片的保存路径（仅当visualization=True时存在）；
  * data 字段为检测结果，类型为list，list的每一个元素为dict，其中'left', 'right', 'top', 'bottom' 为人脸识别框，'confidence' 为此识别框置信度。

```python
def save_inference_model(dirname,
                         model_filename=None,
                         params_filename=None,
                         combined=True)
```

将模型保存到指定路径。

**参数**

* dirname: 存在模型的目录名称
* model_filename: 模型文件名称，默认为\_\_model\_\_
* params_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
* combined: 是否将参数保存到统一的一个文件中

## 预测代码示例

```python
import paddlehub as hub
import cv2

face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
result = face_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = face_detector.face_detection((paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线人脸检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m ultra_light_fast_generic_face_detector_1mb_640
```

这样就完成了一个人脸检测服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

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
url = "http://127.0.0.1:8866/predict/ultra_light_fast_generic_face_detector_1mb_640"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

### 查看代码

https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

### 贡献者

[Jason](https://github.com/jiangjiajun)、[Channingss](https://github.com/Channingss)

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
