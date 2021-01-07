## 概述
* 基于 SINet 模型实现的轻量化人像分割模型
* 模型具体规格如下：
    |model|SINet|
    |----|----|
    |Param|0.087 M|
    |Flop|0.064 G|

* 模型参数转换至 [ext_portrait_segmentation](https://github.com/clovaai/ext_portrait_segmentation) 项目
* 感谢 [ext_portrait_segmentation](https://github.com/clovaai/ext_portrait_segmentation) 项目提供的开源代码和模型

## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/264c689d024942f3817bc9b290dea18812ba88e43d89457e977cd811988f0b44)

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

model = hub.Module(name='SINet_Portrait_Segmentation')

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
