----
# Task
----
在PaddleHub中，Task代表了一个finetune的任务。任务中包含了执行该任务相关的program以及和任务相关的一些度量指标（如准确率accuracy、F1分数）、损失等

## `class paddlehub.finetune.Task(task_type, graph_var_dict, main_program, startup_program)`
> ### 参数
> * task_type: 任务类型，用于在finetune时进行判断如何执行任务
>
> * graph_var_dict: 变量映射表，提供了任务的度量指标
>
> * main_program: 存储了模型计算图的Program
>
> * module_dir: 存储了模型参数初始化op的Program
>
> ### 返回
> Task
>
> ### 示例
>
> ```python
> import paddlehub as hub
> # 根据模型名字创建Module
> resnet = hub.Module(name = "resnet_v2_50_imagenet")
> input_dict, output_dict, program = resnet.context(trainable=True)
> with fluid.program_guard(program):
>     label = fluid.layers.data(name="label", dtype="int64", shape=[1])
>     feature_map = output_dict["feature_map"]
>     task = hub.create_img_cls_task(
>         feature=feature_map, label=label, num_classes=2)
> ```

## `variable(var_name)`
获取Task中的相关变量
> ### 参数
> * var_name: 变量名
>
> ### 示例
>
> ```python
> import paddlehub as hub
> ...
> task = hub.create_img_cls_task(
>     feature=feature_map, label=label, num_classes=2)
> task.variable("loss")
> ```

## `main_program()`
获取Task对应的main_program
> ### 示例
>
> ```python
> import paddlehub as hub
> ...
> task = hub.create_img_cls_task(
>     feature=feature_map, label=label, num_classes=2)
> main_program = task.main_program()
> ```

## `startup_program()`
获取Task对应的startup_program
> ### 示例
>
> ```python
> import paddlehub as hub
> ...
> task = hub.create_img_cls_task(
>     feature=feature_map, label=label, num_classes=2)
> startup_program = task.startup_program()
> ```

## `inference_program()`
获取Task对应的inference_program
> ### 示例
>
> ```python
> import paddlehub as hub
> ...
> task = hub.create_img_cls_task(
>     feature=feature_map, label=label, num_classes=2)
> inference_program = task.inference_program()
> ```

## `metric_variable_names()`
获取Task对应的所有相关的变量，包括loss、度量指标等
> ### 示例
>
> ```python
> import paddlehub as hub
> ...
> task = hub.create_img_cls_task(
>     feature=feature_map, label=label, num_classes=2)
> metric_variable_names = task.metric_variable_names()
> ```
