# Task

在PaddleHub中，Task代表了一个Fine-tune的任务。任务中包含了执行该任务相关的Program、数据Reader、运行配置等内容。

## 基本概念

在了解Task之前，首先需要认识[RunEnv]()和[RunState]()

Task的基本方法和属性参见[BaseTask]()。

## 预置Task

PaddleHub预置了常见任务的Task，每种Task都有自己特有的应用场景以及提供了对应的度量指标，用于适应用户的不同需求。预置的任务类型如下：

* 图像分类任务
[ImageClassifierTask]()
* 文本分类任务
[TextClassifierTask]()
* 序列标注任务
[SequenceLabelTask]()
* 多标签分类任务
[MultiLabelClassifierTask]()
* 回归任务
[RegressionTask]()
* 阅读理解任务
[ReadingComprehensionTask]()

## 自定义Task

如果这些Task不支持您的特定需求，您也可以通过继承BasicTask来实现自己的任务，具体实现细节参见[自定义Task]()

## 修改Task内置方法

如果Task内置方法不满足您的需求，您可以通过Task支持的Hook机制修改方法实现，详细信息参见[修改Task内置方法]()
