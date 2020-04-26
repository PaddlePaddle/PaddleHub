```shell
$ hub install yolov3_darknet53_pedestrian==1.0.0
```

## 命令行预测

```
hub run yolov3_darknet53_pedestrian --input_path "/PATH/TO/IMAGE"
```

## API

```
def context(trainable=True,
            pretrained=True,
            var_prefix='',
            get_prediction=False)
```

提取头部特征，用于迁移学习。

**参数**

* trainable(bool): 将参数的trainable属性设为trainable；
* pretrained (bool): 是否加载预训练模型；
* var\_prefix (str): 在变量的name 中加上前缀；
* get\_prediction (bool): 是否执行预测。

**返回**

* inputs (dict): 模型的输入，keys 包括 'image', 'im\_size'，相应的取值为：
    * image (Variable): 图像变量
    * im\_size (Variable): 图片的尺寸
* outputs (dict): 模型的输出。如果 get\_prediction 为 False，输出 'head\_fatures'，否则输出 'bbox\_out'。
* context\_prog (Program): 用于迁移学习的 Program.

```python
def object_detection(paths=None,
                     images=None,
                     batch_size=1,
                     use_gpu=False,
                     score_thresh=0.2,
                     visualization=True,
                     output_dir='yolov3_pedestrian_detect_output')
```

预测API，检测输入图片中的所有行人的位置。

**参数**

* paths (list\[str\]): 图片的路径；
* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* score\_thresh (float): 识别置信度的阈值；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，默认设为 yolov3\_pedestrian\_detect\_output；

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
    * data (list): 检测结果，list的每一个元素为 dict，各字段为:
        * confidence (float): 识别的置信度；
        * label (str): 标签；
        * left (int): 边界框的左上角x坐标；
        * top (int): 边界框的左上角y坐标；
        * right (int): 边界框的右下角x坐标；
        * bottom (int): 边界框的右下角y坐标；
    * save\_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)。

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

pedestrian_detector = hub.Module(name="yolov3_darknet53_pedestrian")
result = pedestrian_detector.object_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = pedestrian_detector.object_detection((paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个行人检测的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m yolov3_darknet53_pedestrian
```

这样就完成了一个行人检测的服务化API的部署，默认端口号为8866。

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
url = "http://127.0.0.1:8866/predict/yolov3_darknet53_pedestrian"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
