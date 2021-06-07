## 概述
* [SpinalNet](https://arxiv.org/abs/2007.03347)的网络结构如下图，

[网络结构图](https://ai-studio-static-online.cdn.bcebos.com/0c58fff63018401089f92085a2aea5d46921351012e64ac4b7d5a8e1370c463f)

该模型为SpinalNet在宝石数据集上的预训练模型，可以安装PaddleHub后完成一键预测及微调。

## 预训练模型

预训练模型位于https://aistudio.baidu.com/aistudio/datasetdetail/69923

## API
加载该模型后，使用PadduleHub2.0的默认图像分类API
```
def Predict(images, batch_size, top_k):
```

**参数**
* images (list[str: 图片路径]) : 输入图像数据列表
* batch_size: 默认值为1
* top_k: 每张图片的前k个预测类别
