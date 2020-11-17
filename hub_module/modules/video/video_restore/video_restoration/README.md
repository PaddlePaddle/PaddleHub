## 模型概述

video_restoration 是针对老旧视频修复的模型。它主要由三个个部分组成：插帧，着色和超分。插帧模型基于dain模型，着色模型基于deoldify模型，超分模型基于edvr模型. 用户可以根据自己的需求选择对图像进行插帧，着色或超分操作。在使用该模型前请预先安装dain, deoldify以及edvr.


## API

```python
def predict(self,
            input_video_path,
            model_select=['Interpolation', 'Colorization', 'SuperResolution']):
```

预测API，用于视频修复。

**参数**

* input_video_path (str): 视频的路径。

* model_select (list\[str\]): 选择对图片对操作，\['Interpolation'\]对视频只进行插帧操作，\['Colorization'\]对视频只进行着色操作， \['SuperResolution'\]对视频只进行超分操作，
默认值为\['Interpolation', 'Colorization', 'SuperResolution'\]。

**返回**

* temp_video_path (str): 处理后视频保存的位置。



## 代码示例

视频修复代码示例：

```python
import paddlehub as hub

model = hub.Module('video_restoration')
model.predict('/PATH/TO/VIDEO')

```

### 依赖

paddlepaddle >= 2.0.0rc

paddlehub >= 1.8.3
