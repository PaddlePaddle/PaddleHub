## 命令行预测

```
hub run human_pose_estimation_resnet50_mpii --input_path "/PATH/TO/IMAGE"
```

## API 说明

```python
def keypoint_detection(self,
                       images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False,
                       output_dir='output_pose',
                       visualization=False)
```

预测API，识别出人体骨骼关键点。

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，默认设为 output\_pose。

**返回**

* res (list[dict]): 识别元素的列表，列表元素为 dict，关键字为 'path', 'data'，相应的取值为：
    * path (str): 原图的路径；
    * data (OrderedDict): 人体骨骼关键点的坐标。

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

## 预测代码示例

```python
import cv2
import paddlehub as hub

pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")

result = pose_estimation.keypoint_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = pose_estimation.keypoint_detection(paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线人脸关键点检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m human_pose_estimation_resnet50_mpii
```

这样就完成了一个人体骨骼关键点识别的在线服务API的部署，默认端口号为8866。

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
url = "http://127.0.0.1:8866/predict/human_pose_estimation_resnet50_mpii"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```


## 模型相关信息

### 模型代码

https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/human_pose_estimation

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
