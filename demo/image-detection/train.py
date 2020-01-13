# -*- coding:utf8 -*-
import argparse
import os
import ast

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.reader.cv_reader import ObjectDetectionReader
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset
import numpy as np

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch",          type=int,               default=1,                          help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=False,                      help="Whether use GPU for fine-tuning.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt",  help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=2,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="ssd",                 help="Module used as feature extractor.")
parser.add_argument("--dataset",            type=str,               default="coco_10",                  help="Dataset to finetune.")
parser.add_argument("--use_pyreader",       type=ast.literal_eval,  default=False,                      help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel",  type=ast.literal_eval,  default=False,                      help="Whether use data parallel.")
# yapf: enable.

module_map = {
    "yolov3": "yolov3_darknet53_coco2017",
    "ssd": "ssd_vgg16_512_coco2017",
}


def get_model_type(module_name):
    if 'yolo' in module_name:
        return 'yolo'
    elif 'ssd' in module_name:
        return 'ssd'
    elif 'rcnn' in module_map:
        return 'rcnn'
    else:
        raise ValueError("module {} not supported".format(module_name))


def get_feed_list(input_dict, module_name):
    if 'yolo' in module_name:
        img = input_dict["image"]
        im_size = input_dict["im_size"]
        feed_list = [img.name, im_size.name]
    elif 'ssd' in module_map:
        image = input_dict["image"]
        # image_shape = input_dict["im_shape"]
        image_shape = input_dict["im_size"]
        feed_list = [image.name, image_shape.name]
    else:
        raise NotImplementedError
    return feed_list


def get_mid_feature(output_dict, module_name):
    if 'yolo' in module_name:
        feature = output_dict['head_features']
    elif 'ssd' in module_name:
        feature = output_dict['body_features']
    else:
        raise NotImplementedError
    return feature


def finetune(args):
    module_name = args.module  # 'yolov3_darknet53_coco2017'
    model_type = get_model_type(module_name)  # 'yolo'
    # define dataset
    ds = ObjectDetectionDataset(model_type=model_type)
    ds.base_path = '/Users/zhaopenghao/Downloads/coco_10'
    ds.train_image_dir = 'val'
    ds.train_list_file = 'annotations/val.json'
    ds.validate_image_dir = 'val'
    ds.validate_list_file = 'annotations/val.json'
    ds.test_image_dir = 'val'
    ds.test_list_file = 'annotations/val.json'
    # ds.num_labels = 81
    # Todo: handle ds.num_labels refresh
    print(ds.label_dict())
    print("ds.num_labels", ds.num_labels)

    # define batch reader
    data_reader = ObjectDetectionReader(1, 1, dataset=ds, model_type=model_type)

    # define model(program)
    module = hub.Module(name=module_name)
    input_dict, output_dict, program = module.context(trainable=True)
    import pdb; pdb.set_trace()

    feed_list = get_feed_list(input_dict, module_name)
    print("output_dict length:", len(output_dict))
    print(output_dict.keys())
    feature = get_mid_feature(output_dict, module_name)

    config = hub.RunConfig(
        log_interval=1,
        eval_interval=5,
        use_data_parallel=args.use_data_parallel,
        use_pyreader=args.use_pyreader,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    task = hub.DetectionTask(
        data_reader=data_reader,
        feed_list=feed_list,
        feature=feature,
        num_classes=ds.num_labels,
        model_type=model_type,
        config=config)
    task.finetune_and_eval()


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    finetune(args)
