## 命令行预测

```
hub run se_resnet18_vd_imagenet --input_path "/PATH/TO/IMAGE"
```

## API

```python
def get_expected_image_width()
```

返回预处理的图片宽度，也就是224。

```python
def get_expected_image_height()
```

返回预处理的图片高度，也就是224。

```python
def get_pretrained_images_mean()
```

返回预处理的图片均值，也就是 \[0.485, 0.456, 0.406\]。

```python
def get_pretrained_images_std()
```

返回预处理的图片标准差，也就是 \[0.229, 0.224, 0.225\]。


```python
def context(trainable=True, pretrained=True)
```

**参数**

* trainable (bool): 计算图的参数是否为可训练的；
* pretrained (bool): 是否加载默认的预训练模型。

**返回**

* inputs (dict): 计算图的输入，key 为 'image', value 为图片的张量；
* outputs (dict): 计算图的输出，key 为 'classification' 和 'feature_map'，其相应的值为：
    * classification (paddle.fluid.framework.Variable): 分类结果，也就是全连接层的输出；
    * feature\_map (paddle.fluid.framework.Variable): 特征匹配，全连接层前面的那个张量。
* context\_prog(fluid.Program): 计算图，用于迁移学习。

```python
def classification(images=None,
                   paths=None,
                   batch_size=1,
                   use_gpu=False,
                   top_k=1):
```

**参数**

* images (list\[numpy.ndarray\]): 图片数据，每一个图片数据的shape 均为 \[H, W, C\]，颜色空间为 BGR；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU 来预测；
* top\_k (int): 返回预测结果的前 k 个。

**返回**

res (list\[dict\]): 分类结果，列表的每一个元素均为字典，其中 key 为识别动物的类别，value为置信度。

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
import paddlehub as hub
import cv2

classifier = hub.Module(name="se_resnet18_vd_imagenet")

result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = classifier.classification(paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线图像识别服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m se_resnet18_vd_imagenet
```

这样就完成了一个在线图像识别服务化API的部署，默认端口号为8866。

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
url = "http://127.0.0.1:8866/predict/se_resnet18_vd_imagenet"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

### 查看代码

https://github.com/PaddlePaddle/PaddleClas

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
