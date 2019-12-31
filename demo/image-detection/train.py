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


def finetune(args):
    # module = hub.Module(name=args.module)
    # input_dict, output_dict, program = module.context(trainable=True)
    # Todo:
    module_dir = '/Users/zhaopenghao/projects/HubModule/image/object_detection/ssd/v1.0.0/ssd_mobilenet_v1_pascal.hub_module'
    version = 'v3.0.0'
    sig_name = 'feature_map'  # "multi_scale_feature"
    module = hub.Module(module_dir=[module_dir, version])
    input_dict, output_dict, program = module.context(
        trainable=True, sign_name=sig_name)

    # define dataset
    ds = ObjectDetectionDataset()
    ds.base_path = '/Users/zhaopenghao/Downloads/coco_10'
    ds.train_image_dir = 'val'
    ds.train_list_file = 'annotations/val.json'
    ds.validate_image_dir = 'val'
    ds.validate_list_file = 'annotations/val.json'
    ds.test_image_dir = 'val'
    ds.test_list_file = 'annotations/val.json'
    ds.num_labels = 81

    # define batch reader
    data_reader = ObjectDetectionReader(1, 1, dataset=ds)

    print("output_dict", len(output_dict))
    print(output_dict.keys())
    # feature_map = []
    # hub module 重复输出结果两次，
    # for i in range(len(output_dict) // 2):
    #     feature_map.append(output_dict[i])
    fetch_list = [
        'module11', 'module13', 'module14', 'module15', 'module16', 'module17'
    ]
    feature_map = [output_dict[vname] for vname in fetch_list]
    # import pdb; pdb.set_trace()

    img = input_dict["image"]
    feed_list = [img.name]

    config = hub.RunConfig(
        log_interval=1,
        eval_interval=3,
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
        feature=feature_map,
        num_classes=ds.num_labels,
        config=config)
    task.finetune_and_eval()


if __name__ == "__main__":
    args = parser.parse_args()
    # if not args.module in module_map:
    #     hub.logger.error("module should in %s" % module_map.keys())
    #     exit(1)
    # args.module = module_map[args.module]

    finetune(args)
