Extract_Line_Draft
类别 图像 - 图像分割

# 模型概述
提取线稿（Extract_Line_Draft），该模型可自动根据彩色图生成线稿图。该PaddleHub Module支持API预测及命令行预测。

# 选择模型版本进行安装
$ hub install Extract_Line_Draft==1.0.0

# 命令行预测示例
$ hub run Extract_Line_Draft --image 1.png --use_gpu True

# Module API说明
## ExtractLine(self, image, use_gpu=False)
提取线稿预测接口，预测输入一张图像，输出该图像的线稿
### 参数
- image(str): 待检测的图片路径
- use_gpu (bool): 是否使用 GPU


# 代码示例

## API调用
~~~
import paddlehub as hub

Extract_Line_Draft_test = hub.Module(name="Extract_Line_Draft")

test_img = "testImage.png"

# execute predict
Extract_Line_Draft_test.ExtractLine(test_img, use_gpu=True)
~~~

## 命令行调用
~~~
!hub run Extract_Line_Draft --input_path "testImage" --use_gpu True
~~~

# 效果展示

## 原图
![](https://ai-studio-static-online.cdn.bcebos.com/1c30757e069541a18dc89b92f0750983b77ad762560849afa0170046672e57a3)
![](https://ai-studio-static-online.cdn.bcebos.com/4a544c9ecd79461bbc1d1556d100b21d28b41b4f23db440ab776af78764292f2)


## 线稿图
![](https://ai-studio-static-online.cdn.bcebos.com/7ef00637e5974be2847317053f8abe97236cec75fba14f77be2c095529a1eeb3)
![](https://ai-studio-static-online.cdn.bcebos.com/074ea02d89bc4b5c9004a077b61301fa49583c13af734bd6a49e81f59f9cd322)


# 贡献者
彭兆帅、郑博培

# 依赖
paddlepaddle >= 1.8.2
paddlehub >= 1.8.0
