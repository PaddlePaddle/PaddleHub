## 概述
* 基于 face_landmark_localization 和 FCN_HRNet_W18_Face_Seg 模型实现的证件照生成模型，一键生成白底、红底和蓝底的人像照片

## 效果展示
![](https://img-blog.csdnimg.cn/20201224163307901.jpg)

## API
```python
def Photo_GEN(
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False,
        use_gpu=False):
```
证件照生成 API

**参数**
* images (list[np.ndarray]) : 输入图像数据列表（BGR）
* paths (list[str]) : 输入图像路径列表
* batch_size (int) : 数据批大小
* output_dir (str) : 可视化图像输出目录
* visualization (bool) : 是否可视化
* use_gpu (bool) : 是否使用 GPU 进行推理

**返回**
* results (list[dict{"write":np.ndarray,"blue":np.ndarray,"red":np.ndarray}]): 输出图像数据列表

**代码示例**
```python
import cv2
import paddlehub as hub

model = hub.Module(name='ID_Photo_GEN')

result = model.Photo_GEN(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=True,
    use_gpu=False)
```

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
