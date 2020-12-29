## 概述
* ![](http://latex.codecogs.com/svg.latex?U^2Net)的网络结构如下图，其类似于编码-解码(Encoder-Decoder)结构的 U-Net
* 每个 stage 由新提出的 RSU模块(residual U-block) 组成. 例如，En_1 即为基于 RSU 构建的

![](https://ai-studio-static-online.cdn.bcebos.com/999d37b4ffdd49dc9e3315b7cec7b2c6918fdd57c8594ced9dded758a497913d)

## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/4d77bc3a05cf48bba6f67b797978f4cdf10f38288b9645d59393dd85cef58eff)
![](https://ai-studio-static-online.cdn.bcebos.com/865b7b6a262b4ce3bbba4a5c0d973d9eea428bc3e8af4f76a1cdab0c04e3dd33)
![](https://ai-studio-static-online.cdn.bcebos.com/11c9eba8de6d4316b672f10b285245061821f0a744e441f3b80c223881256ca0)

## API
```python
def Segmentation(
        images=None,
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir='output',
        visualization=False):
```
图像前景背景分割 API

**参数**
* images (list[np.ndarray]) : 输入图像数据列表（BGR）
* paths (list[str]) : 输入图像路径列表
* batch_size (int) : 数据批大小
* input_size (int) : 输入图像大小
* output_dir (str) : 可视化图像输出目录
* visualization (bool) : 是否可视化

**返回**
* results (list[np.ndarray]): 输出图像数据列表

**代码示例**
```python
import cv2
import paddlehub as hub

model = hub.Module(name='U2Net')

result = model.Segmentation(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    input_size=320,
    output_dir='output',
    visualization=True)
```

## 查看代码
https://github.com/NathanUA/U-2-Net

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
