## 概述

chinese_ocr_db_rcnn Module用于识别图片当中的汉字。其基于chinese_text_detection_db Module检测得到的文本框，继续识别文本框中的中文文字。识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module支持直接预测。


<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/image/ocr/rcnn.png" hspace='10'/> <br />
</p>

更多详情参考[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)

## 命令行预测

```shell
$ hub run chinese_ocr_db_rcnn --input_path "/PATH/TO/IMAGE"
```

## API

```python
def recognize_texts(paths=[],
                          images=[],
                          use_gpu=False,
                          output_dir='detection_result',
                          box_thresh=0.5,
                          text_thresh=0.5,
                          visualization=False)
```

预测API，检测输入图片中的所有中文文本的位置。

**参数**

* paths (list\[str\]): 图片的路径；
* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**
* box\_thresh (float): 检测文本框置信度的阈值；
* text\_thresh (float): 识别中文文本置信度的阈值；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，默认设为 detection\_result；

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
    * data (list\[dict\]): 识别文本结果，列表中每一个元素为 dict，各字段为：
        * text(str): 识别得到的文本
        * confidence(float): 识别文本结果置信度
        * text_box_position(numpy.ndarray): 文本框在原图中的像素坐标
      如果无识别结果则data为\[\]
    * save_path (str, optional): 识别结果的保存路径，如不保存图片则save_path为\'\'

### 代码示例

```python
import paddlehub as hub
import cv2

ocr = hub.Module(name="chinese_ocr_db_rcnn")

result = ocr.recognize_texts(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = ocr.recognize_texts(paths=['/PATH/TO/IMAGE'])
```

* 样例结果示例

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/image/ocr/ocr_res.png" hspace='10'/> <br />
</p>

## 服务部署

PaddleHub Serving 可以部署一个目标检测的在线服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m chinese_ocr_db_rcnn
```

这样就完成了一个目标检测的服务化API的部署，默认端口号为8866。

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
url = "http://127.0.0.1:8866/predict/chinese_ocr_db_rcnn"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

## 查看代码

https://github.com/PaddlePaddle/PaddleOCR

### 依赖

paddlepaddle >= 1.7.2

paddlehub >= 1.6.0


## 更新历史

* 1.0.0

  初始发布
