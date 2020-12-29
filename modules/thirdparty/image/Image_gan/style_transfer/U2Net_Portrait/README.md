## 概述
* ![](http://latex.codecogs.com/svg.latex?U^2Net) 的网络结构如下图，其类似于编码-解码(Encoder-Decoder)结构的 U-Net
* 每个 stage 由新提出的 RSU模块(residual U-block) 组成. 例如，En_1 即为基于 RSU 构建的
* ![](https://latex.codecogs.com/svg.latex?U^2Net_{Portrait}) 是基于![](http://latex.codecogs.com/svg.latex?U^2Net) 的人脸画像生成模型

![](https://ai-studio-static-online.cdn.bcebos.com/999d37b4ffdd49dc9e3315b7cec7b2c6918fdd57c8594ced9dded758a497913d)

## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/07f73466f3294373965e06c141c4781992f447104a94471dadfabc1c3d920861)
![](https://ai-studio-static-online.cdn.bcebos.com/c6ab02cf27414a5ba5921d9e6b079b487f6cda6026dc4d6dbca8f0167ad7cae3)

## API
```python
def Portrait_GEN(
        images=None,
        paths=None,
        scale=1,
        batch_size=1,
        output_dir='output',
        face_detection=True,
        visualization=False):
```
人脸画像生成 API

**参数**
* images (list[np.ndarray]) : 输入图像数据列表（BGR）
* paths (list[str]) : 输入图像路径列表
* scale (float) : 缩放因子（与face_detection相关联）
* batch_size (int) : 数据批大小
* output_dir (str) : 可视化图像输出目录
* face_detection (bool) : 是否开启人脸检测，开启后会检测人脸并使用人脸中心点进行图像缩放裁切
* visualization (bool) : 是否可视化

**返回**
* results (list[np.ndarray]): 输出图像数据列表

**代码示例**
```python
import cv2
import paddlehub as hub

model = hub.Module(name='U2Net_Portrait')

result = model.Portrait_GEN(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    scale=1,
    batch_size=1,
    output_dir='output',
    face_detection=True,
    visualization=True)
```

## 查看代码
https://github.com/NathanUA/U-2-Net

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
