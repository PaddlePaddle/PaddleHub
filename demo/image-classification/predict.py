#coding:utf-8
import argparse
import os

import paddle.fluid as fluid
import paddlehub as hub
import numpy as np

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu",        type=bool,  default=False,                      help="Whether use GPU for predict.")
parser.add_argument("--checkpoint_dir", type=str,   default="paddlehub_finetune_ckpt",  help="Path to save log data.")
parser.add_argument("--module",         type=str,   default="resnet50",                 help="Module used as a feature extractor.")
parser.add_argument("--dataset",        type=str,   default="flowers",                  help="Dataset to finetune.")
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

    label_map = dataset.label_dict()
    num_labels = len(label_map)

    module = hub.Module(name=args.module)
    input_dict, output_dict, program = module.context()

    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=None)

    img = input_dict["image"]
    feature_map = output_dict["feature_map"]
    task = hub.create_img_cls_task(feature=feature_map, num_classes=num_labels)
    img = input_dict["image"]
    feed_list = [img.name]

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


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    predict(args)
