## 模型概述
edvr是视频超分的模型，该模型基于Video Restoration with Enhanced Deformable Convolutional Networks，可以用于提升老旧视频的分辨率从而提升视频效果。
## API 说明

```python
def predict(self, video_path):
```

补帧API，得到超分后的视频。

**参数**

* video_path (str): 原始视频的路径；

**返回**

* frame_pattern_combined(str): 视频超分后单帧数据保存路径；
* vid_out_path(str): 视频保存路径。


## 预测代码示例

```python
import paddlehub as hub

model = hub.Module('edvr')
model.predict('/PATH/TO//VIDEO')
```

## 模型相关信息

### 模型代码
https://github.com/xinntao/EDVR

### 依赖

paddlepaddle >= 2.0.0rc

paddlehub >= 1.8.3