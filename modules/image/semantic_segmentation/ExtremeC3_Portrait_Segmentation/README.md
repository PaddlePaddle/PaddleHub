## 概述
* 基于 ExtremeC3 模型实现的轻量化人像分割模型
* 模型具体规格如下：
    |model|ExtremeC3|
    |----|----|
    |Param|0.038 M|
    |Flop|0.128 G|

* 模型参数转换至 [ext_portrait_segmentation](https://github.com/clovaai/ext_portrait_segmentation) 项目
* 感谢 [ext_portrait_segmentation](https://github.com/clovaai/ext_portrait_segmentation) 项目提供的开源代码和模型

## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/1261398a98e24184852bdaff5a4e1dbd7739430f59fb47e8b84e3a2cfb976107)

## API
```python
def Segmentation(
    images=None,
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False):
```
人像分割 API

**参数**
* images (list[np.ndarray]) : 输入图像数据列表（BGR）
* paths (list[str]) : 输入图像路径列表
* batch_size (int) : 数据批大小
* output_dir (str) : 可视化图像输出目录
* visualization (bool) : 是否可视化

**返回**
* results (list[dict{"mask":np.ndarray,"result":np.ndarray}]): 输出图像数据列表

**代码示例**
```python
import cv2
import paddlehub as hub

model = hub.Module(name='ExtremeC3_Portrait_Segmentation')

result = model.Segmentation(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False)
```

## 查看代码
https://github.com/clovaai/ext_portrait_segmentation

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
