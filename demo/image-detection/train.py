# -*- coding:utf8 -*-
import argparse
import os
import ast

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.reader.cv_reader import ObjectDetectionReader
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset
import numpy as np
from paddlehub.common.detection_config import get_model_type, get_feed_list, get_mid_feature

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch",          type=int,               default=1,                          help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=False,                      help="Whether use GPU for fine-tuning.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt_yolov3",  help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=1,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="yolov3",                 help="Module used as feature extractor.")
parser.add_argument("--dataset",            type=str,               default="coco_10",                  help="Dataset to finetune.")
parser.add_argument("--use_pyreader",       type=ast.literal_eval,  default=False,                      help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel",  type=ast.literal_eval,  default=False,                      help="Whether use data parallel.")
# yapf: enable.

module_map = {
    "yolov3": "yolov3_darknet53_coco2017",
    "ssd": "ssd_vgg16_512_coco2017",
    "faster_rcnn": "faster_rcnn_resnet50_coco2017",
}


def finetune(args):
    module_name = args.module  # 'yolov3_darknet53_coco2017'
    model_type = get_model_type(module_name)  # 'yolo'
    # define dataset
    ds = hub.dataset.Coco10(model_type)
    # Todo: handle ds.num_labels refresh
    # ds.num_labels = 81
    print(ds.label_dict())
    print("ds.num_labels", ds.num_labels)

    # define batch reader
    # todo: handle img size input arguments
    data_reader = ObjectDetectionReader(1, 1, dataset=ds, model_type=model_type)

    # define model(program)
    module = hub.Module(name=module_name)
    if model_type == 'rcnn':
        input_dict, output_dict, program = module.context(load_pretrained=False, trainable=True, phase='train')
        input_dict_pred, output_dict_pred, program_pred = module.context(trainable=False)
    else:
        input_dict, output_dict, program = module.context(trainable=True)
        input_dict_pred = output_dict_pred = None
    # import pdb; pdb.set_trace()

    feed_list, pred_feed_list = get_feed_list(module_name, input_dict, input_dict_pred)
    print("output_dict length:", len(output_dict))
    print(output_dict.keys())
    feature, pred_feature = get_mid_feature(module_name, output_dict, output_dict_pred)

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
        num_classes=ds.num_labels,
        feed_list=feed_list,
        feature=feature,
        predict_feed_list=pred_feed_list,
        predict_feature=pred_feature,
        model_type=model_type,
        config=config)
    task.finetune_and_eval()
    # self = task
    # with self.phase_guard(phase="train"):
    #     self.init_if_necessary()


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    finetune(args)
