迁移任务
==================

在PaddleHub中，Task代表了一个Fine-tune的任务。任务中包含了执行该任务相关的Program、数据Reader、运行配置等内容。
在了解Task之前，首先需要认识RunEnv和RunState。

Task的基本方法和属性参见BaseTask。

PaddleHub预置了图像分类、文本分类、序列标注、多标签分类、阅读理解、回归等迁移任务，每种任务都有自己特有的应用场景以及提供了对应的度量指标，用于适应用户的不同需求。

..  toctree::
    :maxdepth: 1
    :titlesonly:
    
    hub.task.RunEnv<task/runenv>
    hub.task.RunState<task/runstate>
    hub.task.BaseTask<task/base_task>
    hub.ImageClassifierTask<task/image_classify_task>
    hub.TextClassifierTask<task/text_classify_task>
    hub.SequenceLabelTask<task/sequence_label_task>
    hub.MultiLabelClassifierTask<task/multi_lable_classify_task>
    hub.ReadingComprehensionTask<task/reading_comprehension_task>
    hub.RegressionTask<task/regression_task>