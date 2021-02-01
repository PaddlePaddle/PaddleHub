## 概述
* [SpinalNet](https://arxiv.org/abs/2007.03347)的网络结构如下图，

[网络结构图](https://ai-studio-static-online.cdn.bcebos.com/0c58fff63018401089f92085a2aea5d46921351012e64ac4b7d5a8e1370c463f)

该模型为SpinalNet在宝石数据集上的预训练模型，可以安装PaddleHub后完成一键预测及微调。

## 预训练模型

预训练模型位于https://aistudio.baidu.com/asistudio/datasetdetail/69923

## API
```加载该模型后，使用PadduleHub2.0的默认图像分类API
def Predict(images, batch_size, top_k):
```

**参数**
* images (list[str: 图片路径]) : 输入图像数据列表
* batch_size: 默认值为1
* top_k: 每张图片的前k个预测类别


**返回**
* results (list[dict：{类别:概率}]): 输出类别数据列表

**代码示例**
```python
import paddlehub as hub

model = hub.Module(name='spinalnet_res50_gemstone')
#name的值为spinalnet_res50_gemstone 或者 spinalnet_res101_gemstone 或者 spinalnet_vgg16_gemstone

result = model.predict(
    images=['/PATH/TO/IMAGE'])
```

## 网络结构代码
https://github.com/eepgxxy/SpinalNet_PadddlePaddle

## 依赖
paddlepaddle >= 2.0.0  
paddlehub >= 2.0.0b1
