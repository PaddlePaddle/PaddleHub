# PaddleHub Demo 简介

目前PaddleHub有以下任务示例：

* [图像分类](./image-classification)
  该样例展示了PaddleHub如何将ResNet50、ResNet101、ResNet152、MobileNet、NasNet以及PNasNet作为预训练模型在Flowers、DogCat、Indoor67、Food101、StanfordDogs等数据集上进行图像分类的FineTune和预测。

* [中文词法分析](./lac)
  该样例展示了PaddleHub如何利用中文词法分析LAC进行预测。

* [情感分析](./senta)
  该样例展示了PaddleHub如何利用中文情感分析模型Senta进行FineTune和预测。

* [序列标注](./sequence-labeling)
  该样例展示了PaddleHub如何将ERNIE/BERT等Transformer类模型作为预训练模型在MSRA_NER数据集上完成序列标注的FineTune和预测。

* [目标检测](./ssd)
  该样例展示了PaddleHub如何将SSD作为预训练模型在PascalVOC数据集上完成目标检测的预测。

* [文本分类](./text-classification)
  该样例展示了PaddleHub如何将ERNIE/BERT等Transformer类模型作为预训练模型在GLUE、ChnSentiCorp等数据集上完成文本分类的FineTune和预测。

* [多标签分类](./multi-label-classification)
  该样例展示了PaddleHub如何将BERT作为预训练模型在Toxic数据集上完成多标签分类的FineTune和预测。

* [回归任务](./regression)
  该样例展示了PaddleHub如何将BERT作为预训练模型在GLUE-STSB数据集上完成回归任务的FineTune和预测。

* [阅读理解](./reading-comprehension)
  该样例展示了PaddleHub如何将BERT作为预训练模型在SQAD数据集上完成阅读理解的FineTune和预测。

* [检索式问答任务](./qa_classfication)
  该样例展示了PaddleHub如何将ERNIE和BERT作为预训练模型在NLPCC-DBQA等数据集上完成检索式问答任务的FineTune和预测。

* [句子语义相似度计算](./sentence_similarity)
  该样例展示了PaddleHub如何将word2vec_skipgram用于计算两个文本语义相似度。

* [超参优化AutoDL Finetuner使用](./autofinetune)
  该样例展示了PaddleHub超参优化AutoDL Finetuner如何使用，给出了自动搜素图像分类/文本分类任务的较佳超参数示例。

* [服务化部署Hub Serving使用](./serving)
  该样例文件夹下展示了服务化部署Hub Serving如何使用，将PaddleHub支持的可预测Module如何服务化部署。

**NOTE:**
以上任务示例均是利用PaddleHub提供的数据集，若您想在自定义数据集上完成相应任务，请查看[PaddleHub适配自定义数据完成FineTune](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E9%80%82%E9%85%8D%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E5%AE%8C%E6%88%90FineTune)

## 快速体验

我们在AI Studio上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|ResNet|图像分类|猫狗数据集DogCat|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216772)||
|ERNIE|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216764)||
|ERNIE|文本分类|中文新闻分类数据集THUNEWS|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216649)|本教程讲述了如何将自定义数据集加载，并利用Finetune API完成文本分类迁移学习。|
|ERNIE|序列标注|中文序列标注数据集MSRA_NER|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216787)||
|ERNIE|序列标注|中文快递单数据集Express|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216683)|本教程讲述了如何将自定义数据集加载，并利用Finetune API完成序列标注迁移学习。|
|ERNIE Tiny|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215599)||
|Senta|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216851)|本教程讲述了任何利用Senta和Finetune API完成情感分类迁移学习。|
|Senta|情感分析预测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216735)||
|LAC|词法分析|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215641)||
|Ultra-Light-Fast-Generic-Face-Detector-1MB|人脸检测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216749)||


## 超参优化AutoDL Finetuner

PaddleHub还提供了超参优化（Hyperparameter Tuning）功能， 自动搜索最优模型超参得到更好的模型效果。详细信息参见[AutoDL Finetuner超参优化功能教程](../tutorial/autofinetune.md)
