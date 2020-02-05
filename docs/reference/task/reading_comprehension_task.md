# Class `hub.ReadingComprehensionTask`
阅读理解任务Task，继承自[BaseTask]()，该Task基于输入的特征，添加一个全连接层来创建一个阅读理解任务用于Fine-tune，损失函数为交叉熵Loss。
```python
hub.ReadingComprehensionTask(
    feature,
    feed_list,
    data_reader,
    startup_program=None,
    config=None):
```

**参数**
* feature (fluid.Variable): 输入的特征矩阵。
* feed_list (list): 待feed变量的名字列表
* data_reader: 提供数据的Reader
* startup_program (fluid.Program): 存储了模型参数初始化op的Program，如果未提供，则使用fluid.default_startup_program()
* config ([RunConfig]()): 运行配置

**返回**

`ReadingComprehensionTask`

**示例**

[阅读理解](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.4/demo/reading_comprehension)
