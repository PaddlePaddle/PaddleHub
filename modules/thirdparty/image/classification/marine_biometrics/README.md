MarineBiometrics

类别 图像 - 图像分类

网络 ResNet50_vd_ssld

数据集 Fish4Knowledge

# 模型概述
海洋生物识别（MarineBiometrics），该模型可准确识别鱼的种类。该PaddleHub Module支持API预测及命令行预测。

# 选择模型版本进行安装
$ hub install MarineBiometrics==1.0.0

# 在线体验
[AI Studio快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1667809)

# 命令行预测示例
$ hub run MarineBiometrics --image 1.png --use_gpu True

# Module API说明
## def predict(data)
海洋生物识别预测接口，输入一张图像，输出该图像上鱼的类别
### 参数
- data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。

### 返回
- result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率

# 代码示例

## API调用

~~~
import cv2
import paddlehub as hub

module = hub.Module(name="MarineBiometrics")

images = [cv2.imread('PATH/TO/IMAGE')]

# execute predict and print the result
results = module.predict(images=images)
for result in results:
    print(result)
~~~

## 命令行调用
~~~
$ hub run MarineBiometrics --image 1.png --use_gpu True
~~~

# 效果展示

## 原图
<img src="/docs/imgs/Readme_Related/Image_Classification_MarineBiometrics.png">

## 输出结果
~~~
[{'category_id': 16, 'category': 'Plectroglyphidodon_dickii', 'score': 0.9932127}]
~~~

# 贡献者
郑博培、彭兆帅

# 依赖
paddlepaddle >= 2.0.0

paddlehub >= 2.0.0
