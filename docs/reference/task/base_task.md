# Class `hub.BaseTask`
基础的Task类，封装了finetune、eval、finetune_and_eval、predict等基础接口以及事件的回调机制。该类无法直接使用，需要继承实现特定接口
```python
hub.BaseTask(
    feed_list,
    data_reader,
    main_program=None,
    startup_program=None,
    config=None,
    metrics_choice="default"):
```

**参数**
* feed_list (list): 待feed变量的名字列表
* data_reader: 提供数据的Reader
* main_program (fluid.Program): 存储了模型计算图的Program，如果未提供，则使用fluid.default_main_program()
* startup_program (fluid.Program): 存储了模型参数初始化op的Program，如果未提供，则使用fluid.default_startup_program()
* config ([RunConfig](../config.md)): 运行配置
* metric_choices : 任务评估指标，默认为"acc"。metrics_choices支持训练过程中同时评估多个指标，作为最佳模型的判断依据，例如["matthews", "acc"]，"matthews"将作为主指标，为最佳模型的判断依据；

## 基本概念

### phase / 执行状态

Task可以有不同的执行状态（训练/测试/预测），在不同状态下所获取到的属性会有区别，例如，当处于训练状态时，通过task获取到的feed_list除了输入特征外，还包括输入label，而处于预测状态时，所获取到的feed_list只包括输入特征

Task通过phase属性来区分不同的状态，对应的关系如下：

|phase|状态|
|-|-|
|train|训练|
|val, dev, test|评估|
|predict, inference|预测|

### env / 执行环境
Task中的每个执行状态，都有一个对应的执行环境env（[RunEnv]）用于保存该状态下的属性。当phase发生变化时，env也会发生变化，从而保证用户在不同状态下可以取到正确的属性。

## Func `phase_guard`
配合使用python的“with”语句来改变task的phase状态，在with块中，task的phase为所指定的新phase，退出with块后，恢复上一个phase

**参数**
* phase: 所要切换的phase状态，必须是[有效的phase状态](#phase--执行状态)

**示例**
```python
import paddlehub as hub
...
# 打印该task进行过多少个step的训练
with task.phase_guard("train"):
    print(task.current_step)
```

## Func `enter_phase`
改变task的phase状态

**参数**
* phase: 所要切换的phase状态，必须是[有效的phase状态](#phase--执行状态)

**示例**
```python
import paddlehub as hub
...
# 打印该task进行过多少个step的训练
task.enter_phase("train")
print(task.current_step)
```

## Func `exit_phase`
退出task的当前phase状态，回到上一步的状态

**参数**
* phase: 所要切换的phase状态

**示例**
```python
import paddlehub as hub
...
task.enter_phase("train")
task.exit_phase()
```

## Func `save_checkpoint`
保存当前的checkpoint到config指定的目录

**示例**
```python
import paddlehub as hub
...
with task.phase_guard("train"):
    task.save_checkpoint()
```

## Func `load_checkpoint`
从config指定的checkpoint目录中加载checkpoint数据

**示例**
```python
import paddlehub as hub
...
with task.phase_guard("train"):
    task.load_checkpoint()
```

## Func `save_parameters`
保存参数到指定的目录

**参数**
* dirname: 保存参数的目录

**示例**
```python
import paddlehub as hub
...
with task.phase_guard("train"):
    task.save_parameters("./params")
```

## Func `load_parameters`
从指定的目录中加载参数

**参数**
* dirname: 待加载参数所在目录

**示例**
```python
import paddlehub as hub
...
with task.phase_guard("train"):
    task.load_parameters("./params")
```

## Func `finetune`
根据config配置进行finetune

**示例**
```python
import paddlehub as hub
...
task.finetune()
```

## Func `finetune_and_eval`
根据config配置进行finetune，并且定期进行eval

**示例**
```python
import paddlehub as hub
...
task.finetune_and_eval()
```

## Func `eval`
根据config配置进行eval

**示例**
```python
import paddlehub as hub
...
task.eval(load_best_model = False)
```

## Func `predict`
根据config配置进行predict

**示例**
```python
import paddlehub as hub
...
task.predict()
```


## Property `is_train_phase`
判断是否处于训练阶段

## Property `is_test_phase`
判断是否处于评估阶段

## Property `is_predict_phase`
判断是否处于预测阶段

## Property `phase`
当前的phase

## Property `env`
当前环境[RunEnv]()对象

## Property `py_reader`
当前env中的PyReader对象

## Property `current_step`
当前env所执行过的step数

## Property `current_epoch`
当前env所执行过的epoch数量

## Property `main_program`
当前env对应的主Program，包含训练/评估/预测所需的计算图

## Property `startup_program`
当前env对应的初始化Program，包含初始化op

## Property `reader`
当前env下对应的数据Reader

## Property `loss`
当前env下对应的loss Variable，只在test和train phase有效

## Property `labels`
当前env下对应的label Variable，只在test和train phase有效

## Property `outputs`
当前env下对应的outputs Variable

## Property `metrics`
当前env下对应的metrics Variable，只在test和train phase有效

## Property  `feed_list`
当前env下对应的feed list

## Property `fetch_list`
当前env下对应的fetch_list
