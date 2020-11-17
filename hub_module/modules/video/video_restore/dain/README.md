## 模型概述
dain是视频插帧的模型，该模型基于Depth-Aware Video Frame Interpolation，可以用于老旧视频补帧从而提升视频效果。

## API 说明

```python
def predict(self, video_path):
```

补帧API，得到补帧后的视频。

**参数**

* video_path (str): 原始视频的路径；

**返回**

* frame_pattern_combined(str): 视频补帧后单帧数据保存路径；
* vid_out_path(str): 视频保存路径。


## 预测代码示例

```python
import paddlehub as hub

model = hub.Module('dain')
model.predict('/PATH/TO/VIDEO')
```

## 模型相关信息

### 模型代码
https://github.com/baowenbo/DAIN

### 依赖

paddlepaddle >= 2.0.0rc

paddlehub >= 1.8.3