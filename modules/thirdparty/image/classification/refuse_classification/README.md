refuse_classification

类别 图像 - 图像分类

网络 ResNet50_vd_ssld

# 模型概述
垃圾分类（refuse_classification），该模型可准确识别可回收垃圾、厨余垃圾、有害垃圾和其他垃圾。该PaddleHub Module支持API预测及Serving预测部署。

# 选择模型版本进行安装
$ hub install refuse_classification==1.0.0

# Module API说明

~~~
def predict(data=None,
            batch_size=1,
            use_gpu=False):
~~~

垃圾分类预测接口，输入一张图像，输出该图像上垃圾的类别
### 参数
- data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。
- batch_size：int类型，预测时的batch大小。
- use_gpu：bool类型，是否使用GPU。
### 返回
- res (list[dict]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
    - category_id (int): 类别的id；
    - category（str）: 类别
    - score（float）: 准确率
# 代码示例

## API调用

~~~
import cv2
import paddlehub as hub

module = hub.Module(name="refuse_classification")

images = [cv2.imread('PATH/TO/IMAGE')]

# execute predict and print the result
results = module.predict(images=images)
for result in results:
    print(result)
~~~


# 效果展示

## 原图
<img src="/docs/imgs/Readme_Related/Image_Classification_harmful_garbage.png">

## 输出结果
~~~
[[{'category': 'Harmful', 'category_id': 0, 'score': 0.9257174134254456}]]
~~~

# 贡献者
郑博培、彭兆帅

# 依赖
paddlepaddle >= 2.0.0

paddlehub >= 2.0.0
