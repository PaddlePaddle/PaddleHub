## 概述
* 基于 FCN_HRNet_W18 模型实现的人像分割模型

## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/88155299a7534f1084f8467a4d6db7871dc4729627d3471c9129d316dc4ff9bc)

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
* results (list[dict{"mask":np.ndarray,"face":np.ndarray}]): 输出图像数据列表

**代码示例**
```python
import cv2
import paddlehub as hub

model = hub.Module(name='FCN_HRNet_W18_Face_Seg')

result = model.Segmentation(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=True)
```

## 查看代码
https://github.com/PaddlePaddle/PaddleSeg  
https://github.com/minivision-ai/photo2cartoon-paddle

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
