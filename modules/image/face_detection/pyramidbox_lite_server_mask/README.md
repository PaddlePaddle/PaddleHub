```shell
$ hub install pyramidbox_lite_server_mask==1.3.0
```

## 命令行预测

```
hub run pyramidbox_lite_server_mask --input_path "/PATH/TO/IMAGE"
```

## API

```python
def __init__(face_detector_module=None)
```

**参数**

* face\_detector\_module (class): 人脸检测模型，默认为 pyramidbox\_lite\_server.


```python
def face_detection(images=None,
                   paths=None,
                   batch_size=1,
                   use_gpu=False,
                   visualization=False,
                   output_dir='detection_result',
                   use_multi_scale=False,
                   shrink=0.5,
                   confs_threshold=0.6):
```

识别输入图片中的所有的人脸，并判断有无戴口罩。

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，默认设为 detection\_result。
* use\_multi\_scale (bool) : 用于设置是否开启多尺度的人脸检测，开启多尺度人脸检测能够更好的检测到输入图像中不同尺寸的人脸，但是会增加模型计算量，降低预测速度；
* shrink (float): 用于设置图片的缩放比例，该值越大，则对于输入图片中的小尺寸人脸有更好的检测效果（模型计算成本越高），反之则对于大尺寸人脸有更好的检测效果；
* confs\_threshold (float): 人脸检测的置信度的阈值。

**返回**

* res (list\[dict\]): 识别结果的列表，列表元素为 dict, 有以下两个字段：
    * path (str): 原图的路径。
    * data (list\[dict\]): 识别的边界框列表，有以下字段：
        * label (str): 识别标签，为 'NO MASK' 或者 'MASK'；
        * confidence (float): 识别的置信度；
        * left (int): 边界框的左上角x坐标；
        * top (int): 边界框的左上角y坐标；
        * right (int): 边界框的右下角x坐标；
        * bottom (int): 边界框的右下角y坐标；

```python
def set_face_detector_module(face_detector_module)
```

设置口罩检测模型中进行人脸检测的底座模型。

**参数**

* face\_detector\_module (class): 人脸检测模型

```python
def get_face_detector_module()
```

获取口罩检测模型中进行人脸检测的底座模型。

**返回**

* 当前模型使用的人脸检测模型。

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
import paddlehub as hub
import cv2

mask_detector = hub.Module(name="pyramidbox_lite_server_mask")

result = mask_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = mask_detector.face_detection(paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线人脸关键点检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m pyramidbox_lite_server_mask
```

这样就完成了一个人脸关键点服务化API的部署，默认端口号为8866。

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
url = "http://127.0.0.1:8866/predict/pyramidbox_lite_server_mask"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

## Paddle Lite部署

1.  通过python执行以下代码，保存模型

```python
import paddlehub as hub
pyramidbox_lite_server_mask = hub.Module(name="pyramidbox_lite_server_mask")

# 将模型保存在test_program文件夹之中
pyramidbox_lite_server_mask.save_inference_model(dirname="test_program")
```

通过以上命令，可以获得人脸检测和口罩佩戴判断模型，分别存储在pyramidbox\_lite和mask\_detector之中。文件夹中的\_\_model\_\_是模型结构文件，\_\_params\_\_文件是权重文件。

2.  进行模型转换

从paddlehub下载的是预测模型，可以使用PaddleLite提供的模型优化工具OPT对预测模型进行转换，转换之后进而可以实现在手机等端侧硬件上的部署，具体请请参考[OPT工具](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)

3. 模型通过Paddle Lite进行部署

参考[Paddle-Lite口罩检测模型部署教程](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/cxx)

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
