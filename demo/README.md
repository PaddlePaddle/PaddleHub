# PaddleHub Demo 简介

目前PaddleHub有以下任务示例：

>* 图像分类

>* 中文词法分析

>* 情感分析

>* 序列标注

>* 目标检测

>* 文本分类

>* 多标签分类

>* 检索式问答任务

## 图像分类

该样例展示了PaddleHub如何将ResNet50、ResNet101、ResNet152、MobileNet、NasNet以及PNasNet作为预训练模型在Flowers、DogCat、Indoor67、Food101、StanfordDogs等数据集上进行图像分类的FineTune和预测。

## 中文词法分析

该样例展示了PaddleHub如何利用中文词法分析LAC进行预测。

## 情感分析

该样例展示了PaddleHub如何利用中文情感分析模型Senta进行FineTune和预测。

## 序列标注

该样例展示了PaddleHub如何将ERNIE和BERT作为预训练模型在MSRA_NER数据集上
完成序列标注的FineTune和预测。

## 目标检测

该样例展示了PaddleHub如何将SSD作为预训练模型在PascalVOC数据集上
完成目标检测的预测。

## 文本分类

该样例展示了PaddleHub
>* 如何将ERNIE和BERT作为预训练模型在ChnSentiCorp、LCQMC和NLPCC-DBQA等数据集上完成文本分类的FineTune和预测。  
>* 如何将ELMo预训练得到的中文word embedding加载，完成在ChnSentiCorp数据集上文本分类的FineTune和预测。 

## 多标签分类

该样例展示了PaddleHub如何将BERT作为预训练模型在Toxic数据集上完成多标签分类的FineTune和预测。

## 检索式问答任务

该样例展示了PaddleHub如何将ERNIE和BERT作为预训练模型在NLPCC-DBQA等数据集上完成检索式问答任务的FineTune和预测。


**NOTE**
以上任务示例均是利用PaddleHub提供的数据集，若您想在自定义数据集上完成相应任务，请查看[PaddleHub适配自定义数据完成FineTune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)
