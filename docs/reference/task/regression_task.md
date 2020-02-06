# Class `hub.RegressionTask`

文本回归任务Task，继承自[BaseTask](base_task.md)，该Task基于输入的特征，添加一个Dropout层，以及一个或多个全连接层来创建一个文本回归任务用于finetune，度量指标为准确率，损失函数为均方差损失函数。

```python
hub.RegressionTask(
    feature,
    feed_list,
    data_reader,
    startup_program=None,
    config=None,
    hidden_units=None,
    metrics_choices="default"):
```

**参数**

* feature (fluid.Variable): 输入的特征矩阵。
* feed_list (list): 待feed变量的名字列表。
* data_reader: 提供数据的Reader。
* startup_program (fluid.Program): 存储了模型参数初始化op的Program，如果未提供，则使用fluid.default_startup_program()。
* config ([RunConfig](../config.md)): 运行配置。
* hidden_units (list): RegressionTask最终的全连接层输出维度为1，是一个回归值。在这个全连接层之前可以设置额外的全连接层，并指定它们的输出维度，例如hidden_units=[4,2]表示先经过一层输出维度为4的全连接层，再输入一层输出维度为2的全连接层，最后再输入输出维度为1的全连接层。
* metrics_choices("default" or list ⊂ ["spearman"]): 任务训练过程中需要计算的评估指标，默认为“default”，此时等效于["spearman"]。metrics_choices支持训练过程中同时评估多个指标，其中指定的第一个指标将被作为主指标用于判断当前得分是否为最佳分值，例如["spearman", "acc"]，"spearman"将作为主指标，参与最佳模型的判断中；“acc”只计算并输出，不参与最佳模型的判断。

**返回**

`RegressionTask`

**示例**

[文本回归](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/demo/regression/regression.py)
