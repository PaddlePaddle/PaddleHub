import argparse
import os

import paddle.fluid as fluid
import paddlehub as hub
import numpy as np

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--target",         type=str,   default="finetune",                 help="Number of epoches for fine-tuning.")
parser.add_argument("--num_epoch",      type=int,   default=3,                          help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",        type=bool,  default=False,                      help="Whether use GPU for finetuning or predict")
parser.add_argument("--checkpoint_dir", type=str,   default="paddlehub_finetune_ckpt",  help="Path to training data.")
parser.add_argument("--batch_size",     type=int,   default=16,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",         type=str,   default="resnet50",                 help="Total examples' number in batch for training.")
# yapf: enable.

module_map = {
    "resnet50": "resnet_v2_50_imagenet",
    "resnet101": "resnet_v2_101_imagenet",
    "resnet152": "resnet_v2_152_imagenet",
    "mobilenet": "mobilenet_v2_imagenet",
    "nasnet": "nasnet_imagenet",
    "pnasnet": "pnasnet_imagenet"
}


def get_reader(module, dataset=None):
    return hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=dataset)


def get_task(module, num_classes):
    input_dict, output_dict, program = module.context(trainable=True)
    with fluid.program_guard(program):
        img = input_dict["image"]
        feature_map = output_dict["feature_map"]
        task = hub.create_img_cls_task(
            feature=feature_map, num_classes=num_classes)
    return task


def finetune(args):
    module = hub.Module(name=args.module)
    input_dict, output_dict, program = module.context(trainable=True)
    dataset = hub.dataset.Flowers()
    data_reader = get_reader(module, dataset)
    task = get_task(module, dataset.num_labels)
    img = input_dict["image"]
    feed_list = [img.name, task.variable('label').name]
    config = hub.RunConfig(
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    hub.finetune_and_eval(
        task, feed_list=feed_list, data_reader=data_reader, config=config)


def predict(args):
    module = hub.Module(name=args.module)
    input_dict, output_dict, program = module.context(trainable=True)
    data_reader = get_reader(module)
    task = get_task(module, 5)
    img = input_dict["image"]
    feed_list = [img.name]

    label_map = {
        0: "roses",
        1: "tulips",
        2: "daisy",
        3: "sunflowers",
        4: "dandelion"
    }

    with fluid.program_guard(task.inference_program()):
        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        pretrained_model_dir = os.path.join(args.checkpoint_dir, "best_model")
        if not os.path.exists(pretrained_model_dir):
            hub.logger.error(
                "pretrained model dir %s didn't exist" % pretrained_model_dir)
            exit(1)
        fluid.io.load_persistables(exe, pretrained_model_dir)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        data = ["test/test_img_roses.jpg", "test/test_img_daisy.jpg"]

        predict_reader = data_reader.data_generator(
            phase="predict", batch_size=1, data=data)
        for index, batch in enumerate(predict_reader()):
            result, = exe.run(
                feed=feeder.feed(batch), fetch_list=[task.variable('probs')])
            predict_result = label_map[np.argsort(result[0])[::-1][0]]
            print("input %i is %s, and the predict result is %s" %
                  (index, data[index], predict_result))


def main(args):
    if args.target == "finetune":
        finetune(args)
    elif args.target == "predict":
        predict(args)
    else:
        hub.logger.error("target should in %s" % ["finetune", "predict"])
        exit(1)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    main(args)
