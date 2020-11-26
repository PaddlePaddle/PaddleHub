# PaddleHub 自动数据增强

本示例将展示如何使用PaddleHub搜索最适合数据的数据增强策略，并将其应用到模型训练中。

## 依赖

请预先从pip下载auto-augment软件包

```
pip install -i https://test.pypi.org/simple/ auto-augment
```



## auto-augment简述

auto-augment软件包目前支持Paddle的图像分类任务和物体检测任务。

应用时分成搜索(search)和训练(train)两个阶段

**搜索阶段在预置模型上对不同算子的组合进行策略搜索，输出最优数据增强调度策略组合**

**训练阶段在特定模型上应用最优调度数据增强策略组合 **

详细关于auto-augment的使用及benchmark可参考auto_augment/doc里的readme



## 支持任务

目前auto-augment仅支持paddlhub的图像分类任务。

后续会扩充到其他任务



## 图像分类任务

### 搜索阶段

```
cd PaddleHub/demo/autaug/
bash search
```



## 训练阶段

待完善

