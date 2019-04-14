import paddle.fluid as fluid
import paddlehub as hub

if __name__ == "__main__":
    resnet_module = hub.Module(module_dir="ResNet50.hub_module")
    input_dict, output_dict, program = resnet_module.context(trainable=True)
    dataset = hub.dataset.Flowers()
    data_reader = hub.reader.ImageClassificationReader(
        image_width=resnet_module.get_excepted_image_width(),
        image_height=resnet_module.get_excepted_image_height(),
        images_mean=resnet_module.get_pretrained_images_mean(),
        images_std=resnet_module.get_pretrained_images_std(),
        dataset=dataset)
    with fluid.program_guard(program):
        label = fluid.layers.data(name="label", dtype="int64", shape=[1])
        img = input_dict[0]
        feature_map = output_dict[0]

        config = hub.RunConfig(
            use_cuda=True,
            num_epoch=10,
            batch_size=32,
            enable_memory_optim=False,
            strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

        feed_list = [img.name, label.name]

        task = hub.create_img_cls_task(
            feature=feature_map, label=label, num_classes=dataset.num_labels)
        hub.finetune_and_eval(
            task, feed_list=feed_list, data_reader=data_reader, config=config)
