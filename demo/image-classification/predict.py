#coding:utf-8
import argparse
import os
import ast

import paddle.fluid as fluid
import paddlehub as hub
import numpy as np

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=False,                      help="Whether use GPU for predict.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt",  help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=16,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="resnet50",                 help="Module used as a feature extractor.")
parser.add_argument("--dataset",            type=str,               default="flowers",                  help="Dataset to finetune.")
parser.add_argument("--use_pyreader",       type=ast.literal_eval,  default=False,                      help="Whether use pyreader to feed data.")
# yapf: enable.

module_map = {
    "resnet50": "resnet_v2_50_imagenet",
    "resnet101": "resnet_v2_101_imagenet",
    "resnet152": "resnet_v2_152_imagenet",
    "mobilenet": "mobilenet_v2_imagenet",
    "nasnet": "nasnet_imagenet",
    "pnasnet": "pnasnet_imagenet"
}


def predict(args):
    module = hub.Module(name=args.module)
    input_dict, output_dict, program = module.context(trainable=True)

    if args.dataset.lower() == "flowers":
        dataset = hub.dataset.Flowers()
    elif args.dataset.lower() == "dogcat":
        dataset = hub.dataset.DogCat()
    elif args.dataset.lower() == "indoor67":
        dataset = hub.dataset.Indoor67()
    elif args.dataset.lower() == "food101":
        dataset = hub.dataset.Food101()
    elif args.dataset.lower() == "stanforddogs":
        dataset = hub.dataset.StanfordDogs()
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)

    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=dataset)

    feature_map = output_dict["feature_map"]

    img = input_dict["image"]
    feed_list = [img.name]

    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=args.use_pyreader,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    task = hub.ClassifierTask(
        data_reader=data_reader,
        feed_list=feed_list,
        feature=feature_map,
        num_classes=dataset.num_labels,
        config=config)

    data = ["./test/test_img_daisy.jpg", "./test/test_img_roses.jpg"]
    label_map = dataset.label_dict()
    index = 0
    # get classification result
    run_states = task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    for batch_result in results:
        # get predict index
        batch_result = np.argmax(batch_result, axis=2)[0]
        for result in batch_result:
            index += 1
            result = label_map[result]
            print("input %i is %s, and the predict result is %s" %
                  (index, data[index - 1], result))


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    predict(args)
