# Task
----
在PaddleHub中，Task代表了一个fine-tune的任务。任务中包含了执行该任务相关的program以及和任务相关的一些度量指标（如分类准确率accuracy、precision、 recall、 F1-score等）、模型损失等。

### Class `hub.finetune.task.Task`
```python
hub.Task(
    task_type,
    graph_var_dict,
    main_program,
    startup_program,
    inference_program=None)
```

**参数**
> * task_type: 任务类型，用于在finetune时进行判断如何执行任务
> * graph_var_dict: 变量映射表，提供了任务的度量指标
> * main_program: 存储了模型计算图的Program
> * module_dir: 存储了模型参数初始化op的Program

**返回**

`Task`

**示例**

```python
import paddlehub as hub
# 根据模型名字创建Module
resnet = hub.Module(name = "resnet_v2_50_imagenet")
input_dict, output_dict, program = resnet.context(trainable=True)
feature_map = output_dict["feature_map"]
task = hub.create_img_cls_task(feature=feature_map, num_classes=2)
```

#### `variable`
获取Task的相关变量，包括loss、label以及task相关的性能指标（如分类任务的指标为accuracy）

**参数**
> * var_name: 变量名
>

**示例**
```python
import paddlehub as hub
...
task = hub.create_img_cls_task(
    feature=feature_map, num_classes=2)
task.variable("loss")
```

#### `main_program`
获取Task对应的main_program

**示例**
```python
import paddlehub as hub
...
task = hub.create_img_cls_task(
    feature=feature_map, num_classes=2)
main_program = task.main_program()
```

#### `startup_program`
获取Task对应的startup_program

**示例**
```python
import paddlehub as hub
...
task = hub.create_img_cls_task(
    feature=feature_map, num_classes=2)
startup_program = task.startup_program()
```

#### `inference_program`
获取Task对应的inference_program

**示例**
```python
import paddlehub as hub
...
task = hub.create_img_cls_task(
    feature=feature_map, num_classes=2)
inference_program = task.inference_program()
```

#### `metric_variable_names`
获取Task对应的所有相关的变量，包括loss、度量指标等

**示例**
```python
import paddlehub as hub
...
task = hub.create_img_cls_task(
    feature=feature_map, num_classes=2)
metric_variable_names = task.metric_variable_names()
```


### `hub.create_img_cls_task`

基于输入的特征，添加一个或多个全连接层来创建一个[图像分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/demo/image-classification)任务用于finetune

### 参数
> * feature: 输入的特征
> * labels: 标签Variable
> * num_classes: 最后一层全连接层的神经元个数
> * hidden_units: 隐藏单元的设置，预期值为一个python list，list中的每个元素说明了一个隐藏层的神经元个数

### 返回

`hub.finetune.task.Task`

### 示例

```python
import paddlehub as hub

module = hub.Module(name="resnet_v2_50_imagenet")
inputs, outputs, program = module.context(trainable=True)
feature_map = outputs['feature_map']
cls_task = hub.create_img_cls_task(
        feature=feature_map, num_classes=2, hidden_units = [20, 10])
```

### hub.create_seq_label_task

基于输入的特征，添加一个全连接层来创建一个[序列标注](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/demo/sequence-labeling)任务用于finetune

### 参数
> * feature: 输入的特征
> * seq_len: 序列长度Variable
> * num_classes: 全连接层的神经元个数

### 返回

`hub.finetune.task.Task`

### 示例
```python
import paddlehub as hub

max_seq_len = 20
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=max_seq_len)
sequence_output = outputs["sequence_output"]
seq_label_task = hub.create_seq_label_task(
    feature=sequence_output,
    seq_len=seq_len,
    num_classes=dataset.num_labels)
```

###  hub.create_text_cls_task

基于输入的特征，添加一个或多个全连接层来创建一个[文本分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/demo/text-classification)任务用于finetune

### 参数
> * feature: 输入的特征
> * num_classes: 最后一层全连接层的神经元个数
> * hidden_units: 隐藏单元的设置，预期值为一个python list，list中的每个元素说明了一个隐藏层的神经元个数

### 返回

`hub.finetune.task.Task`

### 示例
```python
import paddlehub as hub

max_seq_len = 20
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=max_seq_len)
pooled_output = outputs["pooled_output"]
cls_task = hub.create_text_cls_task(
    feature=pooled_output, num_classes=2, hidden_units = [20, 10])
```
