## 模型概述
MiDas v2.1 small 单目深度估计模型

模型可通过输入图像估计其中的深度信息

模型权重转换自 [MiDas](https://github.com/intel-isl/MiDaS) 官方开源项目


## 模型安装

```shell
$hub install MiDaS_Small
```

## 效果展示
![效果展示](https://img-blog.csdnimg.cn/20201227112553903.jpg)

## API 说明

```python
def depth_estimation(
    images=None,
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False
)
```

深度估计API

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，默认为 None；
* paths (list\[str\]): 图片的路径，默认为 None；
* batch\_size (int): batch 的大小，默认设为 1；
* visualization (bool): 是否将识别结果保存为图片文件，默认设为 False；
* output\_dir (str): 图片的保存路径，默认设为 output。


**返回**

* res (list\[numpy.ndarray\]): 图像深度数据，ndarray.shape 为 \[H, W\]。


## 预测代码示例

```python
import cv2
import paddlehub as hub

# 模型加载
# use_gpu：是否使用GPU进行预测
model = hub.Module(name='MiDaS_Small', use_gpu=False)

# 模型预测
result = model.depth_estimation(images=[cv2.imread('/PATH/TO/IMAGE')])

# or
# result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
```

## 模型相关信息

### 模型代码

https://github.com/intel-isl/MiDaS

### 依赖

paddlepaddle >= 2.0.0rc0

paddlehub >= 2.0.0b1
