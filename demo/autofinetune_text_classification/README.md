# PaddleHub超参优化——文本分类

**确认安装PaddleHub版本在1.3.0以上, 同时PaddleHub AutoDL Finetuner功能要求至少有一张GPU显卡可用。**

本示例展示如何利用PaddleHub超参优化AutoDL Finetuner，得到一个效果较佳的超参数组合。

每次执行AutoDL Finetuner，用户只需要定义搜索空间，改动几行代码，就能利用PaddleHub搜索最好的超参组合。 只需要两步即可完成：

* 定义搜索空间：AutoDL Finetuner会根据搜索空间来取样生成参数和网络架构。搜索空间通过YAML文件来定义。

* 改动模型代码：需要首先定义参数组，并更新模型代码。

## Step1:定义搜索空间

AutoDL Finetuner会根据搜索空间来取样生成参数和网络架构。搜索空间通过YAML文件来定义。
要定义搜索空间，需要定义变量名称、类型及其搜索范围。通过这些信息构建了一个超参空间，
PaddleHub将在这个空间内进行超参数的搜索，将搜索到的超参传入train.py获得评估效果，根据评估效果自动调整超参搜索方向，直到满足搜索次数。

以Fine-tune文本分类任务为例, 以下是待优化超参数的yaml文件hparam.yaml，包含需要搜素的超参名字、类型、范围等信息。目前参数搜索类型只支持float和int类型。
```
param_list:
- name : learning_rate
  init_value : 0.001
  type : float
  lower_than : 0.05
  greater_than : 0.000005
- name : weight_decay
  init_value : 0.1
  type : float
  lower_than : 1
  greater_than : 0.0
- name : batch_size
  init_value : 32
  type : int
  lower_than : 40
  greater_than : 30
- name : warmup_prop
  init_value : 0.1
  type : float
  lower_than : 0.2
  greater_than : 0.0
```

## Step2:改动模型代码

text_cls.py以ernie为预训练模型，在ChnSentiCorp数据集上进行Fine-tune。PaddleHub如何完成Finetune可以参考[文本分类迁移学习示例](../text_classification)。

* import paddlehub

  在text_cls.py加上`import paddlehub as hub`

* 从AutoDL Finetuner获得参数值

  1. text_cls.py的选项参数须包含待优化超参数，需要将超参以argparser的方式写在其中，待搜索超参数选项名字和yaml文件中的超参数名字保持一致。

  2. text_cls.py须包含选项参数saved_params_dir，优化后的参数将会保存到该路径下。

  3. 超参评估策略选择PopulationBased时，text_cls.py须包含选项参数model_path，自动从model_path指定的路径恢复模型

* 返回配置的最终效果

  text_cls.py须反馈模型的评价效果（建议使用验证集或者测试集上的评价效果），通过调用`report_final_result`接口反馈，如

  ```python
  hub.report_final_result(eval_avg_score["acc"])
  ```

  **NOTE:** 输出的评价效果取值范围应为`(-∞, 1]`，取值越高，表示效果越好。


## 启动AutoDL Finetuner

在完成安装PaddlePaddle与PaddleHub后，通过执行脚本`sh run_autofinetune.sh`即可开始使用超参优化功能。


**NOTE:** 关于PaddleHub超参优化详情参考[教程](../../tutorial/autofinetune.md)。
