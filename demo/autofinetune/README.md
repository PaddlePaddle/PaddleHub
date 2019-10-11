# PaddleHub超参优化——图像分类

本示例展示如何利用PaddleHub超参优化Auto Finetune，得到一个效果较佳的超参数组合

使用PaddleHub Auto Fine-tune需要准备两个指定格式的文件：待优化的超参数信息yaml文件hparam.yaml和需要Fine-tune的python脚本train.py

以Fine-tune图像分类任务为例, 其中：

## hparam.yaml

hparam给出待搜索的超参名字、类型（int或者float，离散型和连续型的两种超参）、搜索范围等信息。
通过这些信息构建了一个超参空间，PaddleHub将在这个空间内进行超参数的搜索，将搜索到的超参传入train.py获得评估效果，根据评估效果自动调整超参搜索方向，直到满足搜索次数。

本示例中待优化超参数为learning_rate和batch_size。


## img_cls.py

以mobilenet为预训练模型，在flowers数据集上进行Fine-tune。


`NOTE`: 关于PaddleHub超参优化详情参考[教程](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.2/tutorial/autofinetune.md)
