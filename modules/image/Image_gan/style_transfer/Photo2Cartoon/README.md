## 概述
* 本模型封装自[小视科技photo2cartoon项目的paddlepaddle版本](https://github.com/minivision-ai/photo2cartoon-paddle)
* 人像卡通风格渲染的目标是，在保持原图像ID信息和纹理细节的同时，将真实照片转换为卡通风格的非真实感图像。我们的思路是，从大量照片/卡通数据中习得照片到卡通画的映射。一般而言，基于成对数据的pix2pix方法能达到较好的图像转换效果，但本任务的输入输出轮廓并非一一对应。例如卡通风格的眼睛更大、下巴更瘦；且成对的数据绘制难度大、成本较高，因此我们采用unpaired image translation方法来实现。模型结构方面，在U-GAT-IT的基础上，我们在编码器之前和解码器之后各增加了2个hourglass模块，渐进地提升模型特征抽象和重建能力。由于实验数据较为匮乏，为了降低训练难度，我们将数据处理成固定的模式。首先检测图像中的人脸及关键点，根据人脸关键点旋转校正图像，并按统一标准裁剪，再将裁剪后的头像输入人像分割模型（基于PaddleSeg框架训练）去除背景。

![](https://ai-studio-static-online.cdn.bcebos.com/8eff9a95bd6741beb3895f38eca39265f22c358c7d114c11b400bbbcd9c4cfc0)

## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/a4aaedc5ede449e282f0a1c1df05566b62737ddec98246a9b2d5cfeb0f005563)

## API
```python
def Cartoon_GEN(
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False,
        use_gpu=False):
```
人像卡通化图像生成 API

**参数**
* images (list[np.ndarray]) : 输入图像数据列表（BGR）
* paths (list[str]) : 输入图像路径列表
* batch_size (int) : 数据批大小
* output_dir (str) : 可视化图像输出目录
* visualization (bool) : 是否可视化
* use_gpu (bool) : 是否使用 GPU 进行推理

**返回**
* results (list[np.ndarray]): 输出图像数据列表

**代码示例**
```python
import cv2
import paddlehub as hub

model = hub.Module(name='Photo2Cartoon')

result = model.Cartoon_GEN(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=True,
    use_gpu=False)
```

## 查看代码
https://github.com/PaddlePaddle/PaddleSeg  
https://github.com/minivision-ai/photo2cartoon-paddle

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
