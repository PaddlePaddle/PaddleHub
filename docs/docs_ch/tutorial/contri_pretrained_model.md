# 如何编写一个PaddleHub Module

## 模型基本信息

我们准备编写一个PaddleHub Module，Module的基本信息如下：
```yaml
name="openpose_body_estimation",
type="CV/image_editing",
author="paddlepaddle",
author_email="",
summary="Openpose_body_estimation is a body pose estimation model based on Realtime Multi-Person 2D Pose \
        Estimation using Part Affinity Fields.",
version="1.0.0"

```

Module存在一个接口predict，用于接收传入图片，并得到最终输出的结果，支持python接口调用和命令行调用。
```python
import paddlehub as hub

model = hub.Module(name="openpose_body_estimation")
result = model.predict("demo.jpg")
```
```cmd
hub run openpose_body_estimation --input_path demo.jpg
```


## Module创建

### step 1. 创建必要的目录与文件

创建一个openpose_body_estimation的目录，并在openpose_body_estimation目录下分别创建module.py, processor.py。其中

|文件名|用途|
|-|-|
|module.py|主模块，提供Module的实现代码|
|processor.py|辅助模块，提供词表加载的方法|

```cmd
➜  tree openpose_body_estimation
openpose_body_estimation/
   ├── module.py
   └── processor.py
```
### step 2. 实现辅助模块processor

在processor.py中实现一些在module.py里面需要调用到的类和函数。例如在processor.py 中实现ResizeScaling类：

```python
class ResizeScaling:
    """Resize images by scaling method.

    Args:
        target(int): Target image size.
        interpolation(Callable): Interpolation method.
    """

    def __init__(self, target: int = 368, interpolation: Callable = cv2.INTER_CUBIC):
        self.target = target
        self.interpolation = interpolation

    def __call__(self, img, scale_search):
        scale = scale_search * self.target / img.shape[0]
        resize_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=self.interpolation)
        return resize_img
```

### step 3. 编写Module处理代码

module.py文件为Module的入口代码所在，我们需要在其中实现预测逻辑。

#### step 3_1. 引入必要的头文件
```python
import os
import time
import copy
import base64
import argparse
from typing import Union
from collections import OrderedDict

import cv2
import paddle
import paddle.nn as nn
import numpy as np
from paddlehub.module.module import moduleinfo, runnable, serving
import paddlehub.vision.transforms as T
import openpose_body_estimation.processor as P
```
**NOTE:** `paddlehub.vision.transforms`有常见的图像处理方法，可以方便调用。

#### step 3_2. 定义BodyPoseModel类
module.py中需要有一个继承了nn.Layer，该类负责实现预测逻辑，并使用moduleinfo填写基本信息。当使用hub.Module(name="openpose_body_estimation")加载Module时，PaddleHub会自动创建openpose_body_estimation的对象并返回。
```python
@moduleinfo(
    name="openpose_body_estimation",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="Openpose_body_estimation is a body pose estimation model based on Realtime Multi-Person 2D Pose \
            Estimation using Part Affinity Fields.",
    version="1.0.0")
class BodyPoseModel(nn.Layer):
    ...
```
#### step 3_3. 执行必要的初始化及模型搭建
模型的初始化主要完成几个功能：待使用的类的声明，模型使用的类的声明及参数加载。
```python
 def __init__(self, load_checkpoint: str = None):
        super(BodyPoseModel, self).__init__()
        #将会使用到的类的声明
        self.resize_func = P.ResizeScaling()
        self.norm_func = T.Normalize(std=[1, 1, 1])
        #模型声明
        self.input_nc = 4
        self.output_nc = 2
        model1 = (
            Conv2D(self.input_nc, 64, 3, 1, 1),
            nn.ReLU(),
            Conv2D(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(64),
        )
        self.model1 = nn.Sequential(*model1)
        #参数加载
        if load_checkpoint is not None:
            self.model_dict = paddle.load(load_checkpoint)
            self.set_dict(self.model_dict)
            print("load custom checkpoint success")
        else:
            checkpoint = os.path.join(self.directory, 'model.pdparams')
            self.model_dict = paddle.load(checkpoint)
            self.set_dict(self.model_dict)
            print("load pretrained checkpoint success")

```  
模型的搭建主要在`forward`里面实现：
```python
def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        result = self.model1(input)
        return result

```  

#### step 3_4. 完善预测逻辑
```python
def predict(self, img:Union(np.ndarray,str), visualization: bool = True):
    self.eval()
    self.visualization = visualization
    if isinstance(img, str):
        orgImg = cv2.imread(img)
    else:
        orgImg = img
    data = self.resize_func(self.norm_func(orgImg))
    output = self.forward(paddle.to_tensor(data.astype('float32')))
    output = paddle.clip(output[0].transpose((1, 2, 0)), 0, 255).numpy()
    output = output.astype(np.uint8)
    if self.visualization:
        style_name = "body_" + str(time.time()) + ".png"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        path = os.path.join(save_path, style_name)
        cv2.imwrite(path, output)
    return output
```
#### step 3_5. 支持命令行调用
如果希望Module可以支持命令行调用，则需要提供一个经过runnable修饰的接口，接口负责解析传入数据并进行预测，将结果返回。

```python
@runnable
def run_cmd(self, argvs):
    """
    Run as a command.
    """
    self.parser = argparse.ArgumentParser(
        description="Run the {} module.".format(self.name),
        prog='hub run {}'.format(self.name),
        usage='%(prog)s',
        add_help=True)
    self.arg_input_group = self.parser.add_argument_group(
        title="Input options", description="Input data. Required")
    self.arg_config_group = self.parser.add_argument_group(
        title="Config options",
        description=
        "Run configuration for controlling module behavior, not required.")
    self.add_module_config_arg()
    self.add_module_input_arg()
    args = self.parser.parse_args(argvs)
    results = self.predict(
        img=args.input_path,
        save_path=args.output_dir,
        visualization=args.visualization)
    return results

def add_module_config_arg(self):
    """
    Add the command config options.
    """

    self.arg_config_group.add_argument(
        '--output_dir',
        type=str,
        default='openpose_body',
        help="The directory to save output images.")
    self.arg_config_group.add_argument(
        '--save_dir',
        type=str,
        default='openpose_model',
        help="The directory to save model.")
    self.arg_config_group.add_argument(
        '--visualization',
        type=bool,
        default=True,
        help="whether to save output as images.")

def add_module_input_arg(self):
    """
    Add the command input options.
    """
    self.arg_input_group.add_argument(
        '--input_path', type=str, help="path to image.")

```
#### step 3_6. 支持serving调用

如果希望Module可以支持PaddleHub Serving部署预测服务，则需要提供一个经过serving修饰的接口，接口负责解析传入数据并进行预测，将结果返回。

如果不需要提供PaddleHub Serving部署预测服务，则可以不需要加上serving修饰。

```python
@serving
def serving_method(self, images, **kwargs):
    """
    Run as a service.
    """
    images_decode = [base64_to_cv2(image) for image in images]
    results = self.predict(img=images_decode[0], **kwargs)
    final={}
    final['data'] = P.cv2_to_base64(results)
    return final
```


## 测试步骤

完成Module编写后，我们可以通过以下方式测试该Module

### 调用方法1

将Module安装到本机中，再通过Hub.Module(name=...)加载
```shell
hub install openpose_body_estimation
```

```python
import paddlehub as hub

if __name__ == "__main__":

    model = hub.Module(name='openpose_hands_estimation')
    result = model.predict("demo.jpg")
```

### 调用方法2
将Module安装到本机中，再通过hub run运行

```shell
hub install openpose_body_estimation
hub run openpose_body_estimation --input_path demo.jpg
```
### 测试serving方法

运行启动命令：

```shell
$ hub serving start -m openpose_body_estimation
```

发送预测请求，获取预测结果.

```python
import requests
import json
import cv2
import base64

import numpy as np


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

# 发送HTTP请求
org_im = cv2.imread('/PATH/TO/IMAGE')
data = {'images':[cv2_to_base64(org_im)]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/openpose_body_estimation"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
canvas = base64_to_cv2(r.json()["results"]['data'])
cv2.imwrite('keypoint_body.png', canvas)
```
