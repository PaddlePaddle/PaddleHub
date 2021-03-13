DriverStatusRecognition
类别 图像 - 图像分类
网络 MobileNetV3_small_ssld
数据集 分心司机检测数据集

# 模型概述
驾驶员状态识别（DriverStatusRecognition），该模型可挖掘出人在疲劳状态下的表情特征，然后将这些定性的表情特征进行量化，提取出面部特征点及特征指标作为判断依据，再结合实验数据总结出基于这些参数的识别方法，最后输入获取到的状态数据进行识别和判断。该PaddleHub Module支持API预测及命令行预测。

# 选择模型版本进行安装
$ hub install DriverStatusRecognition==1.0.0

# 在线体验
[AI Studio快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1649513)

# 命令行预测示例
$ hub run DriverStatusRecognition --image 1.png --use_gpu True

# Module API说明
## def predict(data)
驾驶员状态识别预测接口，输入一张图像，输出该图像上驾驶员的状态
### 参数
- data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。

### 返回
- result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率

# 代码示例

## API调用
~~~
import cv2
import paddlehub as hub

module = hub.Module(directory='DriverStatusRecognition') # 一行代码实现模型调用

images = [cv2.imread('work/imgs/test/img_1622.jpg'), cv2.imread('work/imgs/test/img_14165.jpg'), cv2.imread('work/imgs/test/img_47183.jpg')]
results = module.predict(images=images)

for result in results:
    print(result)
~~~

## 命令行调用
~~~
$ hub run DriverStatusRecognition --image 1.png --use_gpu True
~~~

# 效果展示

## 原图
![](https://ai-studio-static-online.cdn.bcebos.com/da3d4ca593c94d8f9fb96d5aa81523ebd7c6ea6dbfa24853831db51fb0098a5e)

## 输出结果
~~~
[{'category_id': 5, 'category': 'ch5', 'score': 0.47390476}]
[{'category_id': 2, 'category': 'ch2', 'score': 0.99997914}]
[{'category_id': 1, 'category': 'ch1', 'score': 0.99996376}]
~~~

# 贡献者
郑博培、彭兆帅

# 依赖
paddlepaddle >= 2.0.0
paddlehub >= 2.0.0