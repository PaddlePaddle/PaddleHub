# finetune_and_eval
----

## `method paddlehub.finetune.task.finetune_and_eval(task, data_reader, feed_list, config=None):`

对一个Task进行finetune，并且定期进行验证集评估。在finetune的过程中，接口会定期的保存checkpoint（模型和运行数据），当运行被中断时，通过RunConfig指定上一次运行的checkpoint目录，可以直接从上一次运行的最后一次评估中恢复状态继续运行
> ### 参数
> * task: 需要执行的Task
>
> * data_reader: 提供数据的reader
>
> * feed_list: reader的feed列表
>
> * config: 运行配置
>
> ### 示例
>
> ```python
> import paddlehub as hub
> import paddle.fluid as fluid
>
> resnet_module = hub.Module(name="resnet_v2_50_imagenet")
> input_dict, output_dict, program = resnet_module.context(trainable=True)
> dataset = hub.dataset.Flowers()
> data_reader = hub.reader.ImageClassificationReader(
>     image_width=resnet_module.get_excepted_image_width(),
>     image_height=resnet_module.get_excepted_image_height(),
>     dataset=dataset)

> img = input_dict["image"]
> 
> feature_map = output_dict["feature_map"]
>
> feed_list = [img.name, label.name]
>
> task = hub.create_img_cls_task(
>     feature=feature_map, num_classes=dataset.num_labels)
> hub.finetune_and_eval(
>     task=task, feed_list=feed_list, data_reader=data_reader)
> ```
