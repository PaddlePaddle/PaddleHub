----
# create_text_cls_task
----

## `method paddlehub.finetune.task.create_text_cls_task(feature, label, num_classes, hidden_units=None):`

基于输入的特征，添加一个或多个全连接层来创建一个[文本分类](https://github.com/PaddlePaddle/PaddleHub/tree/develop/demo/text-classification)任务用于finetune
> ### 参数
> * feature: 输入的特征
>
> * labels: 标签Variable
>
> * num_classes: 最后一层全连接层的神经元个数
>
> * hidden_units: 隐藏单元的设置，预期值为一个python list，list中的每个元素说明了一个隐藏层的神经元个数
>
> ### 返回
> paddle.finetune.task.Task
>
> ### 示例
>
> ```python
> import paddlehub as hub
>
> max_seq_len = 20
> module = hub.Module(name="ernie")
> inputs, outputs, program = module.context(
>     trainable=True, max_seq_len=max_seq_len)
>
> with fluid.program_guard(program):
>     label = fluid.layers.data(name="label", shape=[1], dtype='int64')
>     pooled_output = outputs["pooled_output"]
>
>     cls_task = hub.create_text_cls_task(
>         feature=pooled_output, label=label, num_classes=2, hidden_units = [20, 10])
> ```
