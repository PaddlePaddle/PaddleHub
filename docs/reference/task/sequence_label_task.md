# Class `hub.SequenceLabelTask`
序列标注Task，继承自[BaseTask](base_task.md)，该Task基于输入的特征，添加一个全连接层或者一个全连接层和CRF层来创建一个序列标注任务用于Fine-tune，度量指标为F1，损失函数为交叉熵Loss。
```python
hub.SequenceLabelTask(
    feature,
    max_seq_len,
    num_classes,
    feed_list,
    data_reader,
    startup_program=None,
    config=None,
    metrics_choices="default",
    add_crf=False):
```

**参数**
* feature (fluid.Variable): 输入的特征矩阵
* num_classes (int): 分类任务的类别数量
* feed_list (list): 待feed变量的名字列表
* data_reader: 提供数据的Reader
* startup_program (fluid.Program): 存储了模型参数初始化op的Program，如果未提供，则使用fluid.default_startup_program()
* config ([RunConfig](../config.md)): 运行配置
* metrics_choices("default" or list ⊂ ["precision","recall","f1"]): 任务训练过程中需要计算的评估指标，默认为“default”，此时等效于["f1", "precision", "recall"]。metrics_choices支持训练过程中同时评估多个指标，其中指定的第一个指标将被作为主指标用于判断当前得分是否为最佳分值，例如["f1", "precision"]，"f1"将作为主指标，参与最佳模型的判断中；"precision"只计算并输出，不参与最佳模型的判断
* add_crf (bool): 是否选择crf作为decoder，默认为false，如果add_crf=True，则网络加入crf层作为decoder
**返回**

`SequenceLabelTask`

**示例**

[序列标注](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/demo/sequence_labeling/sequence_label.py)
