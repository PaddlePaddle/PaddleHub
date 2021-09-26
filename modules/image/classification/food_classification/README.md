# food_classification

类别 图像 - 图像分类

网络 ResNet50_vd_ssld


> 模型概述

美食分类（food_classification），该模型可识别苹果派，小排骨，烤面包，牛肉馅饼，牛肉鞑靼。该PaddleHub Module支持API预测及命令行预测。

> 选择模型版本进行安装

```shell
$ hub install food_classification==1.0.0
```
> Module API说明

```python
def predict(self,
                images=None,
                paths=None,
                batch_size=1,
                use_gpu=False,
                **kwargs):
```
美食分类预测接口，输入一张图像，输出该图像上食物的类别

参数

* images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
* paths (list[str]): 图片的路径；
* batch_size (int): batch 的大小；
* use_gpu (bool): 是否使用 GPU；

返回

* res (list[dict]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
    * category_id (int): 类别的id；
    * category（str）: 类别;
    * score（float）: 准确率;

## 代码示例

### API调用

```python
import cv2
import paddlehub as hub

module = hub.Module(name="food_classification")

images = [cv2.imread('PATH/TO/IMAGE')]

# execute predict and print the result
results = module.predict(images=images)
for result in results:
    print(result)
```

### 命令行调用
```shell
$ hub run food_classification --input_path /PATH/TO/IMAGE --use_gpu True
```

## 效果展示

### 原图
<img src="/docs/imgs/Readme_Related/Image_Classification_apple_pie.png">

### 输出结果
```python
[{'category_id': 0, 'category': 'apple_pie', 'score': 0.9985085}]
```

## 贡献者
彭兆帅、郑博培

## 依赖
paddlepaddle >= 2.0.0

paddlehub >= 2.0.0

paddlex >= 1.3.7
