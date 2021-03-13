SnakeIdentification
类别 图像 - 图像分类
网络 ResNet50_vd_ssld
数据集 蛇种数据集

# 模型概述
蛇种识别（SnakeIdentification），该模型可准确识别蛇的种类，并精准判断蛇的毒性。该PaddleHub Module支持API预测及命令行预测。

# 选择模型版本进行安装
$ hub install SnakeIdentification==1.0.0

# 在线体验
[AI Studio快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1646951)

# 命令行预测示例
$ hub run SnakeIdentification --image 1.png --use_gpu True

# Module API说明
## def predict(data)
蛇种识别预测接口，输入一张图像，输出该图像上蛇的类别
### 参数
- data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。

### 返回
- result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率

# 代码示例

## API调用
~~~
import cv2
import paddlehub as hub

module = hub.Module(name="SnakeIdentification")

images = [cv2.imread('snake_data/class_1/2421.jpg')]

# execute predict and print the result
results = module.predict(images=images)
for result in results:
    print(result)
~~~

## 命令行调用
~~~
$ hub run SnakeIdentification --image 1.png --use_gpu True
~~~

# 效果展示

## 原图
![](https://ai-studio-static-online.cdn.bcebos.com/818dcf6d44554137b835c1b8f07a86a6ff2688da2b2e44cb984c7a4e61ceacbe)

## 输出结果
~~~
[{'category_id': 0, 'category': '水蛇', 'score': 0.9999205}]
~~~

# 贡献者
郑博培、彭兆帅

# 依赖
paddlepaddle >= 2.0.0
paddlehub >= 2.0.0