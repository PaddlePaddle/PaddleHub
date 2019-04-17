----
# create_seq_label_task
----

## `method paddlehub.finetune.task.create_seq_label_task(feature, labels, seq_len, num_classes)`

基于输入的特征，添加一个全连接层来创建一个[序列标注](https://github.com/PaddlePaddle/PaddleHub/tree/develop/demo/sequence-labeling)任务用于finetune
> ### 参数
> * feature: 输入的特征
>
> * labels: 标签Variable
>
> * seq_len: 序列长度Variable
>
> * num_classes: 全连接层的神经元个数
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
>     label = fluid.layers.data(name="label", shape=[max_seq_len, 1], dtype='int64')
>     seq_len = fluid.layers.data(name="seq_len", shape=[1], dtype='int64')
>     sequence_output = outputs["sequence_output"]
>
>     seq_label_task = hub.create_seq_label_task(
>          feature=sequence_output,
>          labels=label,
>          seq_len=seq_len,
>          num_classes=dataset.num_labels)
> ```
