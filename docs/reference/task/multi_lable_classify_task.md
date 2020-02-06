# Class `hub.MultiLabelClassifierTask`
多标签分类Task，继承自[BaseTask](base_task.md)，该Task基于输入的特征，添加一个或多个全连接层来创建一个多标签分类任务用于finetune，度量指标为多个标签的平均AUC，损失函数为多个标签的平均交叉熵。
```python
hub.MultiLabelClassifierTask(
    feature,
    num_classes,
    feed_list,
    data_reader,
    startup_program=None,
    config=None,
    hidden_units=None,
    metrics_choices="default"):
```

**参数**
* feature (fluid.Variable): 输入的特征矩阵。
* num_classes (int): 多标签任务的标签数量
* feed_list (list): 待feed变量的名字列表
* data_reader: 提供数据的Reader
* startup_program (fluid.Program): 存储了模型参数初始化op的Program，如果未提供，则使用fluid.default_startup_program()
* config ([RunConfig](../config.md)): 运行配置
* hidden_units (list): MultiLabelClassifierTask最终的全连接层输出维度为[num_classes, 2]，是属于各个标签的概率值。在这个全连接层之前可以设置额外的全连接层，并指定它们的输出维度，例如hidden_units=[4，2]表示先经过一层输出维度为4的全连接层，再输入一层输出维度为2的全连接层，最后再拼接上输出维度为[num_classes, 2]的全连接层。
* metrics_choices("default" or list ⊂ ["auc"]): 任务训练过程中需要计算的评估指标，默认为“default”，此时等效于["auc"]。

**返回**

`MultiLabelClassifierTask`

**示例**

[多标签分类](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/demo/multi_label_classification/multi_label_classifier.py)
