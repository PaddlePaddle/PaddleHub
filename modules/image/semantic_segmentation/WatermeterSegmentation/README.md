WatermeterSegmentation
类别 图像 - 图像分割
网络 DeepLabV3
数据集 水表的数字表盘分割数据集

# 模型概述
水表的数字表盘分割（WatermeterSegmentation），该模型可自动提取水表上的数字。该PaddleHub Module支持API预测及命令行预测。

# 选择模型版本进行安装
$ hub install WatermeterSegmentation==1.0.0

# 在线体验
[AI Studio快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1643214)

# 命令行预测示例
$ hub run WatermeterSegmentation --image 1.png --use_gpu True

# Module API说明
## def cutPic(picUrl)
水表的数字表盘分割预测接口，输入一张图像，输出该图像上水表记录的数字
### 参数
- picUrl(str): 待检测的图片路径

# 代码示例

## API调用
~~~
import cv2
import paddlehub as hub

seg = hub.Module(name='WatermeterSegmentation')
res = seg.cutPic(picUrl="1.png")
~~~

## 命令行调用
~~~
$ hub run WatermeterSegmentation --image 1.png --use_gpu True
~~~

# 效果展示

## 原图
<img src="/docs/imgs/Readme_Related/ImageSeg_WaterInput.png">

## 输出结果
<img src="/docs/imgs/Readme_Related/ImageSeg_WaterOutput.png">

# 贡献者
郑博培、彭兆帅

# 依赖
paddlepaddle >= 2.0.0<br>
paddlehub >= 2.0.0
