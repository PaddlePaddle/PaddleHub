## 模型概述

barometer_reader是基于PaddleX实现对传统机械式指针表计的检测与自动读数功能的模型。该模型首先使用目标检测模型检测出图像中的表计，随后使用语义分割模型将各表计的指针和刻度分割，最后根据指针的相对位置和预知的量程计算出各表计的读数。

## 命令行预测

```
$ hub run barometer_reader --input_path "/PATH/TO/IMAGE"

```

## API

```python
def predict(self,
            im_file: Union[str, np.ndarray],
            score_threshold: float = 0.5,
            seg_batch_size: int = 2,
            erode_kernel: int = 4,
            use_erode: bool = True,
            visualization: bool = False,
            save_dir: str ='output'):
```

预测API，用于表针读数。

**参数**

* im_file (Union\[str, np.ndarray\]): 图片路径或者图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* score\_threshold (float): 检测模型输出结果中，预测得分低于该阈值的框将被滤除，默认值为0.5；
* seg\_batch\_size (int): 分割的批量大小，默认为2；
* erode\_kernel (int): 图像腐蚀操作时的卷积核大小，默认值为4；
* use\_erode (str): 是否使用图像腐蚀对分割预测图进行细分，默认为False;
* visualization (bool): 是否将可视化图片保存；
* save_dir (str): 保存图片到路径， 默认为"output"。

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 'category\_id', 'bbox', 'score','category', 对应的取值为：
  * category\_id (int): bounding box框出的图片的类别号；
  * bbox (list): bounding box数值；
  * score (float): bbox类别得分；
  * category (str):  bounding box框出的图片的类别名称。


## 代码示例

```python
import cv2
import paddlehub as hub

model = hub.Module(name='barometer_reader')
res = model.predict('/PATH/TO/IMAGE')
print(res)
```

## 服务部署

PaddleHub Serving可以部署一个表计识别的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m barometer_reader
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


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


if __name__ == '__main__':
    # 获取图片的base64编码格式
    img = cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))
    data = {'image': img}
    # 指定content-type
    headers = {"Content-type": "application/json"}
    # 发送HTTP请求
    url = "http://127.0.0.1:8866/predict/barometer_reader"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    # 打印预测结果
    print(r.json())
```

### 查看代码

https://github.com/PaddlePaddle/PaddleX


### 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.0.0

paddlex >= 1.3.0
